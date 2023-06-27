"""
General layers used in the architectures
"""
from torch.nn import Parameter as P

import torch
import torch.nn as nn
import torch.nn.functional as F

activation_dict = {
    "inplace_relu": nn.ReLU(inplace=True),
    "relu": nn.ReLU(inplace=False),
    "ir": nn.ReLU(inplace=True),
}

# Simple wrapper that applies EMA to a model. COuld be better done in 1.0 using
# the parameters() and buffers() module functions, but for now this works
# with state_dicts using .copy_
class ema(object):
    def __init__(self, source, target, decay=0.9999, start_itr=0):
        self.source = source
        self.target = target
        self.decay = decay
        # Optional parameter indicating what iteration to start the decay at
        self.start_itr = start_itr
        # Initialize target's params to be source's
        self.source_dict = self.source.state_dict()
        self.target_dict = self.target.state_dict()
        print("Initializing EMA parameters to be source parameters...")
        with torch.no_grad():
            for key in self.source_dict:
                self.target_dict[key].data.copy_(self.source_dict[key].data)
                # target_dict[key].data = source_dict[key].data # Doesn't work!

    def update(self, itr=None):
        # If an iteration counter is provided and itr is less than the start itr,
        # peg the ema weights to the underlying weights.
        if itr and itr < self.start_itr:
            decay = 0.0
        else:
            decay = self.decay
        with torch.no_grad():
            for key in self.source_dict:
                self.target_dict[key].data.copy_(
                    self.target_dict[key].data * decay
                    + self.source_dict[key].data * (1 - decay)
                )


# Projection of x onto y
def proj(x, y):
    return torch.mm(y, x.t()) * y / torch.mm(y, y.t())


# Orthogonalize x wrt list of vectors ys
def gram_schmidt(x, ys):
    for y in ys:
        x = x - proj(x, y)
    return x


# Apply num_itrs steps of the power method to estimate top N singular values.
def power_iteration(W, u_, update=True, eps=1e-12):
    # Lists holding singular vectors and values
    us, vs, svs = [], [], []
    for i, u in enumerate(u_):
        # Run one step of the power iteration
        with torch.no_grad():
            v = torch.matmul(u, W)
            # Run Gram-Schmidt to subtract components of all other singular vectors
            v = F.normalize(gram_schmidt(v, vs), eps=eps)
            # Add to the list
            vs += [v]
            # Update the other singular vector
            u = torch.matmul(v, W.t())
            # Run Gram-Schmidt to subtract components of all other singular vectors
            u = F.normalize(gram_schmidt(u, us), eps=eps)
            # Add to the list
            us += [u]
            if update:
                u_[i][:] = u
        # Compute this singular value and add it to the list
        svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
        # svs += [torch.sum(F.linear(u, W.transpose(0, 1)) * v)]
    return svs, us, vs


# Convenience passthrough function
class identity(nn.Module):
    def forward(self, input):
        return input


# Spectral normalization base class
class SN(object):
    def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
        # Number of power iterations per step
        self.num_itrs = num_itrs
        # Number of singular values
        self.num_svs = num_svs
        # Transposed?
        self.transpose = transpose
        # Epsilon value for avoiding divide-by-0
        self.eps = eps
        # Register a singular vector for each sv
        for i in range(self.num_svs):
            self.register_buffer("u%d" % i, torch.randn(1, num_outputs))
            self.register_buffer("sv%d" % i, torch.ones(1))

    # Singular vectors (u side)
    @property
    def u(self):
        return [getattr(self, "u%d" % i) for i in range(self.num_svs)]

    # Singular values;
    # note that these buffers are just for logging and are not used in training.
    @property
    def sv(self):
        return [getattr(self, "sv%d" % i) for i in range(self.num_svs)]

    # Compute the spectrally-normalized weight
    def W_(self):
        W_mat = self.weight.view(self.weight.size(0), -1)
        if self.transpose:
            W_mat = W_mat.t()
        # Apply num_itrs power iterations
        for _ in range(self.num_itrs):
            svs, us, vs = power_iteration(
                W_mat, self.u, update=self.training, eps=self.eps
            )
        # Update the svs
        if self.training:
            with torch.no_grad():  # Make sure to do this in a no_grad() context or you'll get memory leaks!
                for i, sv in enumerate(svs):
                    self.sv[i][:] = sv
        return self.weight / svs[0]


# 2D Conv layer with spectral norm
class SNConv2d(nn.Conv2d, SN):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        num_svs=1,
        num_itrs=1,
        eps=1e-12,
    ):
        nn.Conv2d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)

    def forward(self, x):
        return F.conv2d(
            x,
            self.W_(),
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


# Linear layer with spectral norm
class SNLinear(nn.Linear, SN):
    def __init__(
        self, in_features, out_features, bias=True, num_svs=1, num_itrs=1, eps=1e-12
    ):
        nn.Linear.__init__(self, in_features, out_features, bias)
        SN.__init__(self, num_svs, num_itrs, out_features, eps=eps)

    def forward(self, x):
        return F.linear(x, self.W_(), self.bias)


# Embedding layer with spectral norm
# We use num_embeddings as the dim instead of embedding_dim here
# for convenience sake
class SNEmbedding(nn.Embedding, SN):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        padding_idx=None,
        max_norm=None,
        norm_type=2,
        scale_grad_by_freq=False,
        sparse=False,
        _weight=None,
        num_svs=1,
        num_itrs=1,
        eps=1e-12,
    ):
        nn.Embedding.__init__(
            self,
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            _weight,
        )
        SN.__init__(self, num_svs, num_itrs, num_embeddings, eps=eps)

    def forward(self, x):
        return F.embedding(x, self.W_())


# A non-local block as used in SA-GAN
# Note that the implementation as described in the paper is largely incorrect;
# refer to the released code for the actual implementation.
class Attention(nn.Module):
    def __init__(self, ch, which_conv=SNConv2d, name="attention"):
        super(Attention, self).__init__()
        # Channel multiplier
        self.ch = ch
        self.which_conv = which_conv
        self.theta = self.which_conv(
            self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False
        )
        self.phi = self.which_conv(
            self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False
        )
        self.g = self.which_conv(
            self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False
        )
        self.o = self.which_conv(
            self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False
        )
        # Learnable gain parameter
        self.gamma = P(torch.tensor(0.0), requires_grad=True)

    def forward(self, x, y=None):
        # Apply convs
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2, 2])
        g = F.max_pool2d(self.g(x), [2, 2])
        # Perform reshapes
        theta = theta.view(-1, self.ch // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self.ch // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self.ch // 2, x.shape[2] * x.shape[3] // 4)
        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(
            torch.bmm(g, beta.transpose(1, 2)).view(
                -1, self.ch // 2, x.shape[2], x.shape[3]
            )
        )
        return self.gamma * o + x


# Simple function to handle groupnorm norm stylization
def groupnorm(x, norm_style):
    # If number of channels specified in norm_style:
    if "ch" in norm_style:
        ch = int(norm_style.split("_")[-1])
        groups = max(int(x.shape[1]) // ch, 1)
    # If number of groups specified in norm style
    elif "grp" in norm_style:
        groups = int(norm_style.split("_")[-1])
    # If neither, default to groups = 16
    else:
        groups = 16
    return F.group_norm(x, groups)


# Class-conditional bn
# output size is the number of channels, input size is for the linear layers
# Andy's Note: this class feels messy but I'm not really sure how to clean it up
# Suggestions welcome! (By which I mean, refactor this and make a pull request
# if you want to make this more readable/usable).
class ccbn(nn.Module):
    def __init__(
        self,
        output_size,
        input_size,
        which_linear,
        eps=1e-5,
        momentum=0.1,
        norm_style="bn",
    ):
        super(ccbn, self).__init__()
        self.output_size, self.input_size = output_size, input_size
        # Prepare gain and bias layers
        self.gain = which_linear(input_size, output_size)
        self.bias = which_linear(input_size, output_size)
        # epsilon to avoid dividing by 0
        self.eps = eps
        # Momentum
        self.momentum = momentum
        # Norm style?
        self.norm_style = norm_style

        if self.norm_style in ["bn", "in"]:
            self.register_buffer("stored_mean", torch.zeros(output_size))
            self.register_buffer("stored_var", torch.ones(output_size))

    def forward(self, x, y):
        # Calculate conditional gains and biases
        gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
        bias = self.bias(y).view(y.size(0), -1, 1, 1)
        if self.norm_style == "bn":
            out = F.batch_norm(
                x,
                self.stored_mean,
                self.stored_var,
                None,
                None,
                self.training,
                0.1,
                self.eps,
            )
        elif self.norm_style == "in":
            out = F.instance_norm(
                x,
                self.stored_mean,
                self.stored_var,
                None,
                None,
                self.training,
                0.1,
                self.eps,
            )
        elif self.norm_style == "gn":
            out = groupnorm(x, self.normstyle)
        elif self.norm_style == "nonorm":
            out = x
        return out * gain + bias

    def extra_repr(self):
        s = "out: {output_size}, in: {input_size},"
        return s.format(**self.__dict__)


# Normal, non-class-conditional BN
class bn(nn.Module):
    def __init__(self, output_size, norm_style="bn", eps=1e-5, momentum=0.1):
        super(bn, self).__init__()
        self.output_size = output_size
        # Prepare gain and bias layers
        self.gain = P(torch.ones(output_size), requires_grad=True)
        self.bias = P(torch.zeros(output_size), requires_grad=True)
        # epsilon to avoid dividing by 0
        self.eps = eps
        # Momentum
        self.momentum = momentum
        # Norm style?
        self.norm_style = norm_style

        self.register_buffer("stored_mean", torch.zeros(output_size))
        self.register_buffer("stored_var", torch.ones(output_size))

    def forward(self, x, y=None):
        if self.norm_style == "bn":
            return F.batch_norm(
                x,
                self.stored_mean,
                self.stored_var,
                self.gain,
                self.bias,
                self.training,
                self.momentum,
                self.eps,
            )
        elif self.norm_style == "in":
            return F.instance_norm(
                x,
                self.stored_mean,
                self.stored_var,
                self.gain,
                self.bias,
                self.training,
                0.1,
                self.eps,
            )


class ccbn_1d(nn.Module):
    def __init__(
        self,
        output_size,
        input_size,
        which_linear,
        eps=1e-5,
        momentum=0.1,
        norm_style="bn",
    ):
        super(ccbn, self).__init__()
        self.output_size, self.input_size = output_size, input_size
        # Prepare gain and bias layers
        self.gain = which_linear(input_size, output_size)
        self.bias = which_linear(input_size, output_size)
        # epsilon to avoid dividing by 0
        self.eps = eps
        # Momentum
        self.momentum = momentum
        # Norm style?
        self.norm_style = norm_style

        self.register_buffer("stored_mean", torch.zeros(output_size))
        self.register_buffer("stored_var", torch.ones(output_size))

    def forward(self, x, y):
        # Calculate class-conditional gains and biases
        gain = (1 + self.gain(-y)).view(y.size(0), -1)
        bias = self.bias(y).view(y.size(0), -1)
        # If using my batchnorm
        if self.norm_style == "bn":
            out = F.batch_norm(
                x,
                self.stored_mean,
                self.stored_var,
                None,
                None,
                self.training,
                0.1,
                self.eps,
            )
        elif self.norm_style == "in":
            out = F.instance_norm(
                x,
                self.stored_mean,
                self.stored_var,
                None,
                None,
                self.training,
                0.1,
                self.eps,
            )
        elif self.norm_style == "gn":
            out = groupnorm(x, self.normstyle)
        elif self.norm_style == "nonorm":
            out = x
        return out * gain + bias

    def extra_repr(self):
        s = "out: {output_size}, in: {input_size},"
        return s.format(**self.__dict__)


# Normal, non-class-conditional BN
class bn_1d(nn.Module):
    def __init__(self, output_size, eps=1e-5, momentum=0.1):
        super(bn_1d, self).__init__()
        self.output_size = output_size
        # Prepare gain and bias layers
        self.gain = P(torch.ones(output_size), requires_grad=True)
        self.bias = P(torch.zeros(output_size), requires_grad=True)
        # epsilon to avoid dividing by 0
        self.eps = eps
        # Momentum
        self.momentum = momentum

        self.register_buffer("stored_mean", torch.zeros(output_size))
        self.register_buffer("stored_var", torch.ones(output_size))

    def forward(self, x, y=None):
        return F.batch_norm(
            x,
            self.stored_mean,
            self.stored_var,
            self.gain,
            self.bias,
            self.training,
            self.momentum,
            self.eps,
        )


class SmallMLP(nn.Module):
    def __init__(
        self,
        input_dim=0,
        output_dim=0,
        activation=nn.ReLU(inplace=False),
        which_linear=nn.Linear,
        **kwargs
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            which_linear(input_dim, output_dim),
            Transpose(1, 2),
            bn_1d(output_dim),
            Transpose(1, 2),
            activation,
        )

    def forward(self, x):
        return self.mlp(x)


class Sharpen(nn.Module):
    def __init__(self, is_sharpen):
        super(Sharpen, self).__init__()
        self.is_sharpen = is_sharpen

    def forward(self, x):
        if self.is_sharpen:
            return x / self.is_sharpen
        else:
            return x


class unMaxPool(nn.Module):
    def __init__(self, num_reap=2):
        super(unMaxPool, self).__init__()
        self.num_reap = num_reap

    def forward(self, x):
        return x.repeat_interleave(self.num_reap, dim=2)


class minPool(nn.Module):
    def __init__(self, scale=2):
        super(minPool, self).__init__()
        self.max = nn.MaxPool2d(scale, 0)

    def forward(self, x):
        return -self.max(-x)


class Reshape(nn.Module):
    def __init__(self, dataset_channel=1, output_dim=6):
        super(Reshape, self).__init__()
        self.datset_channel = dataset_channel
        self.output_dim = output_dim

    def forward(self, x):
        return x.reshape(-1, self.datset_channel, self.output_dim)


class Transpose(nn.Module):
    def __init__(self, index1, index2):
        super(Transpose, self).__init__()
        self.index1 = index1
        self.index2 = index2

    def forward(self, x):
        return x.transpose(self.index1, self.index2)
        # return x


class parAct(nn.Module):
    def __init__(self, type, num_prot=1, bias=1e-6):
        super(parAct, self).__init__()
        self.type = type[:3]
        self.pond = 1
        self.num_prot = num_prot
        if not self.type == "exp":
            self.pond = float(type[3:])
        self.bias = bias
        if self.pond == -1:
            self.pond = 1 / (2 * num_prot)
            # self.pond = 1/(4*num_prot)

    def update_pond(self, time_max=1):
        self.pond = time_max * (1 / (2 * self.num_prot))
        # self.pond = time_max * (1 / (4 * self.num_prot))

    def forward(self, x):
        if self.type == "exp":
            return x.exp() + self.bias
        elif self.type == "sig":
            return self.pond * x.sigmoid() + self.bias
        elif self.type == "tnh":
            return self.pond * x.tanh()
