import torch

from diff_svc.modules.commons.common_layers import *
from diff_svc.modules.commons.common_layers import Embedding
from diff_svc.modules.commons.common_layers import SinusoidalPositionalEmbedding
from diff_svc.utils.hparams import hparams
from diff_svc.utils.pitch_utils import f0_to_coarse, denorm_f0


class LayerNorm(torch.nn.LayerNorm):
    """Layer normalization module.
    :param int nout: output dim size
    :param int dim: dimension to be normalized
    """

    def __init__(self, nout, dim=-1):
        """Construct an LayerNorm object."""
        super(LayerNorm, self).__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.
        :param torch.Tensor x: input tensor
        :return: layer normalized tensor
        :rtype torch.Tensor
        """
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return super(LayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1)


class PitchPredictor(torch.nn.Module):
    def __init__(self, idim, n_layers=5, n_chans=384, odim=2, kernel_size=5,
                 dropout_rate=0.1, padding='SAME'):
        """Initilize pitch predictor module.
        Args:
            idim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
        """
        super(PitchPredictor, self).__init__()
        self.conv = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        self.padding = padding
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(
                torch.nn.ConstantPad1d(((kernel_size - 1) // 2, (kernel_size - 1) // 2)
                                       if padding == 'SAME'
                                       else (kernel_size - 1, 0), 0),
                torch.nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=0),
                torch.nn.ReLU(),
                LayerNorm(n_chans, dim=1),
                torch.nn.Dropout(dropout_rate)
            )]
        self.linear = torch.nn.Linear(n_chans, odim)
        self.embed_positions = SinusoidalPositionalEmbedding(idim, 0, init_size=4096)
        self.pos_embed_alpha = nn.Parameter(torch.Tensor([1]))

    def forward(self, xs):
        """

        :param xs: [B, T, H]
        :return: [B, T, H]
        """
        positions = self.pos_embed_alpha * self.embed_positions(xs[..., 0])
        xs = xs + positions
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)
        # NOTE: calculate in log domain
        xs = self.linear(xs.transpose(1, -1))  # (B, Tmax, H)
        return xs


class SvcEncoder(nn.Module):
    def __init__(self, dictionary, out_dims=None):
        super().__init__()
        # self.dictionary = dictionary
        self.padding_idx = 0
        self.hidden_size = hparams['hidden_size']
        self.out_dims = out_dims
        if out_dims is None:
            self.out_dims = hparams['audio_num_mel_bins']
        self.mel_out = Linear(self.hidden_size, self.out_dims, bias=True)
        predictor_hidden = hparams['predictor_hidden'] if hparams['predictor_hidden'] > 0 else self.hidden_size
        if hparams['use_pitch_embed']:
            self.pitch_embed = Embedding(300, self.hidden_size, self.padding_idx)
            self.pitch_predictor = PitchPredictor(
                self.hidden_size,
                n_chans=predictor_hidden,
                n_layers=hparams['predictor_layers'],
                dropout_rate=hparams['predictor_dropout'],
                odim=2 if hparams['pitch_type'] == 'frame' else 1,
                padding=hparams['ffn_padding'], kernel_size=hparams['predictor_kernel'])
        if hparams['use_energy_embed']:
            self.energy_embed = Embedding(256, self.hidden_size, self.padding_idx)
        if hparams['use_spk_id']:
            self.spk_embed_proj = Embedding(hparams['num_spk'], self.hidden_size)
        elif hparams['use_spk_embed']:
            self.spk_embed_proj = Linear(256, self.hidden_size, bias=True)

    def forward(self, hubert, mel2ph=None, spk_embed_id=None, f0=None, energy=None):
        ret = {}
        encoder_out = hubert
        var_embed = 0

        # encoder_out_dur denotes encoder outputs for duration predictor
        # in speech adaptation, duration predictor use old speaker embedding
        if hparams['use_spk_id']:
            spk_embed_0 = self.spk_embed_proj(spk_embed_id.to(hubert.device))[:, None, :]
            spk_embed_1 = self.spk_embed_proj(torch.LongTensor([0]).to(hubert.device))[:, None, :]
            spk_embed_2 = self.spk_embed_proj(torch.LongTensor([0]).to(hubert.device))[:, None, :]
            spk_embed = 1 * spk_embed_0 + 0 * spk_embed_1 + 0 * spk_embed_2
            spk_embed_f0 = spk_embed
        else:
            spk_embed_f0 = spk_embed = 0

        ret['mel2ph'] = mel2ph

        decoder_inp = F.pad(encoder_out, [0, 0, 1, 0])

        mel2ph_ = mel2ph[..., None].repeat([1, 1, encoder_out.shape[-1]])
        decoder_inp_origin = decoder_inp = torch.gather(decoder_inp, 1, mel2ph_)  # [B, T, H]

        tgt_nonpadding = (mel2ph > 0).float()[:, :, None]

        # add pitch and energy embed
        pitch_inp = (decoder_inp_origin + var_embed + spk_embed_f0) * tgt_nonpadding
        if hparams['use_pitch_embed']:
            decoder_inp = decoder_inp + self.add_pitch(pitch_inp, f0, mel2ph, ret)
        if hparams['use_energy_embed']:
            decoder_inp = decoder_inp + self.add_energy(pitch_inp, energy, ret)

        ret['decoder_inp'] = (decoder_inp + spk_embed) * tgt_nonpadding
        return ret

    def add_pitch(self, decoder_inp, f0, mel2ph, ret):
        decoder_inp = decoder_inp.detach() + hparams['predictor_grad'] * (decoder_inp - decoder_inp.detach())
        pitch_padding = (mel2ph == 0)
        ret['f0_denorm'] = f0_denorm = denorm_f0(f0, False, hparams, pitch_padding=pitch_padding)
        if pitch_padding is not None:
            f0[pitch_padding] = 0

        pitch = f0_to_coarse(f0_denorm, hparams)  # start from 0
        ret['pitch_pred'] = pitch.unsqueeze(-1)
        pitch_embedding = self.pitch_embed(pitch)
        return pitch_embedding

    def add_energy(self, decoder_inp, energy, ret):
        decoder_inp = decoder_inp.detach() + hparams['predictor_grad'] * (decoder_inp - decoder_inp.detach())
        ret['energy_pred'] = energy  # energy_pred = self.energy_predictor(decoder_inp)[:, :, 0]
        energy = torch.clamp(energy * 256 // 4, max=255).long()  # energy_to_coarse
        energy_embedding = self.energy_embed(energy)
        return energy_embedding
