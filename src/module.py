"""Torch modules for my graph attention networks."""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn

from dgl import function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair
from dgl.nn.functional import edge_softmax
from dgl.nn.pytorch.utils import Identity


class EdgeGATv2Conv(nn.Module):
    r"""GATv2 from `How Attentive are Graph Attention Networks?
    <https://arxiv.org/pdf/2105.14491.pdf>`__

    .. math::
        h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{ij}^{(l)} W^{(l)}_{right} h_j^{(l)}

    where :math:`\alpha_{ij}` is the attention score bewteen node :math:`i` and
    node :math:`j`:

    .. math::
        \alpha_{ij}^{(l)} &= \mathrm{softmax_i} (e_{ij}^{(l)})

        e_{ij}^{(l)} &= {\vec{a}^T}^{(l)}\mathrm{LeakyReLU}\left(
            W^{(l)}_{left} h_{i} + W^{(l)}_{right} h_{j}\right)
    """

    def __init__(
        self,
        in_feats,
        out_feats,
        edge_feats,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        bias=True,
        share_weights=False,
    ):
        super(EdgeGATv2Conv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=bias
            )
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=bias
            )
        else:
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=bias
            )
            if share_weights:
                self.fc_dst = self.fc_src
            else:
                self.fc_dst = nn.Linear(
                    self._in_src_feats, out_feats * num_heads, bias=bias
                )
        self.attn = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=bias
                )
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer("res_fc", None)
        
        self._edge_feats = edge_feats
        self.fc_edge = nn.Linear(edge_feats, out_feats * num_heads, bias=bias)
        self.attn_edge = nn.Parameter(
            th.FloatTensor(size=(1, num_heads, out_feats))
        )
        self.activation = activation
        self.share_weights = share_weights
        self.bias = bias
        self.reset_parameters()

    def reset_parameters(self):

        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
        if self.bias:
            nn.init.constant_(self.fc_src.bias, 0)
        if not self.share_weights:
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
            if self.bias:
                nn.init.constant_(self.fc_dst.bias, 0)
        
        nn.init.xavier_normal_(self.attn, gain=gain)

        nn.init.xavier_normal_(self.fc_edge.weight, gain=gain)
        if self.bias:
            nn.init.constant_(self.fc_edge.bias, 0)
        nn.init.xavier_normal_(self.attn_edge, gain=gain)

        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
            if self.bias:
                nn.init.constant_(self.res_fc.bias, 0)

    def set_allow_zero_in_degree(self, set_value):

        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, edge_feat, get_attention=False):
        
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                feat_src = self.fc_src(h_src).view(
                    -1, self._num_heads, self._out_feats
                )
                feat_dst = self.fc_dst(h_dst).view(
                    -1, self._num_heads, self._out_feats
                )
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = self.fc_src(h_src).view(
                    -1, self._num_heads, self._out_feats
                )
                if self.share_weights:
                    feat_dst = feat_src
                else:
                    feat_dst = self.fc_dst(h_dst).view(
                        -1, self._num_heads, self._out_feats
                    )
                if graph.is_block:
                    feat_dst = feat_dst[: graph.number_of_dst_nodes()]
                    h_dst = h_dst[: graph.number_of_dst_nodes()]
            # Linearly tranform the edge features.
            n_edges = edge_feat.shape[:-1]
            feat_edge = self.fc_edge(edge_feat).view(
                *n_edges, self._num_heads, self._out_feats
            )
            # Add edge features to graph.
            # graph.edata["ft_edge"] = feat_edge
            graph.edata["ee"] = feat_edge

            graph.srcdata.update(
                {"el": feat_src}
            )  # (num_src_edge, num_heads, out_dim)
            graph.dstdata.update({"er": feat_dst})
            graph.apply_edges(fn.u_add_v("el", "er", "e_tmp"))

            # e_tmp combines attention weights of source and destination node.
            # Add the attention weight of the edge.
            graph.edata["e"] = graph.edata["e_tmp"] + graph.edata["ee"]
            # Create new edges features that combine the
            # features of the source node and the edge features.
            graph.apply_edges(fn.u_add_e("el", "ee", "ft_combined"))

            e = self.leaky_relu(
                graph.edata.pop("e")
            )  # (num_src_edge, num_heads, out_dim)
            e = (
                (e * self.attn).sum(dim=-1).unsqueeze(dim=2)
            )  # (num_edge, num_heads, 1)
            # compute softmax
            graph.edata["a"] = self.attn_drop(
                edge_softmax(graph, e)
            )  # (num_edge, num_heads)

            # For each edge, element-wise multiply the combined features with
            # the attention coefficient.
            graph.edata["m_combined"] = (
                graph.edata["ft_combined"] * graph.edata["a"]
            )

            # First copy the edge features and then sum them up.
            graph.update_all(fn.copy_e("m_combined", "m"), fn.sum("m", "ft"))

            # message passing
            # graph.update_all(fn.u_mul_e("el", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(
                    h_dst.shape[0], -1, self._out_feats
                )
                rst = rst + resval
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata["a"]
            else:
                return rst


# pylint: enable=W0235
class EdgeGATConv(nn.Module):
    r"""Graph attention layer with edge features from `SCENE
    <https://arxiv.org/pdf/2301.03512.pdf>`__

    .. math::

        \mathbf{v}_i^\prime = \mathbf{\Theta}_\mathrm{s} \cdot \mathbf{v}_i +
        \sum\limits_{j \in \mathcal{N}(v_i)} \alpha_{j, i} \left( \mathbf{\Theta}_\mathrm{n}
        \cdot \mathbf{v}_j + \mathbf{\Theta}_\mathrm{e} \cdot \mathbf{e}_{j,i} \right)

    where :math:`\mathbf{\Theta}` is used to denote learnable weight matrices
    for the transformation of features of the node to update (s=self),
    neighboring nodes (n=neighbor) and edge features (e=edge).
    Attention weights are obtained by

    .. math::

        \alpha_{j, i} = \mathrm{softmax}_i \Big( \mathrm{LeakyReLU} \big( \mathbf{a}^T
        [ \mathbf{\Theta}_\mathrm{n} \cdot \mathbf{v}_i || \mathbf{\Theta}_\mathrm{n}
        \cdot \mathbf{v}_j || \mathbf{\Theta}_\mathrm{e} \cdot \mathbf{e}_{j,i} ] \big) \Big)

    with :math:`\mathbf{a}` corresponding to a learnable vector.
    :math:`\mathrm{softmax_i}` stands for the normalization by all incoming edges of node :math:`i`.
    """

    def __init__(
        self,
        in_feats,
        edge_feats,
        out_feats,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=True,
        activation=None,
        allow_zero_in_degree=False,
        bias=True,
    ):
        super(EdgeGATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False
            )
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False
            )
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False
            )
        self.attn_l = nn.Parameter(
            th.FloatTensor(size=(1, num_heads, out_feats))
        )
        self.attn_r = nn.Parameter(
            th.FloatTensor(size=(1, num_heads, out_feats))
        )
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(
                th.FloatTensor(size=(num_heads * out_feats,))
            )
        else:
            self.register_buffer("bias", None)
        if residual:
            self.res_fc = nn.Linear(
                self._in_dst_feats, num_heads * out_feats, bias=False
            )
        else:
            self.register_buffer("res_fc", None)

        self._edge_feats = edge_feats
        self.fc_edge = nn.Linear(edge_feats, out_feats * num_heads, bias=False)
        self.attn_edge = nn.Parameter(
            th.FloatTensor(size=(1, num_heads, out_feats))
        )

        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`\mathbf{\Theta}` are and the
        attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)

        nn.init.xavier_normal_(self.fc_edge.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_edge, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, edge_feat, get_attention=False):
        r"""

        Description
        -----------
        Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, *, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, *, D_{in_{src}})` and :math:`(N_{out}, *, D_{in_{dst}})`.
        edge_feat : torch.Tensor
            The input edge feature of shape :math:`(E, D_{in_{edge}})`,
            where :math:`E` is the number of edges and :math:`D_{in_{edge}}`
            the size of the edge features.
        get_attention : bool, optional
            Whether to return the attention values. Default to False.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, *, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        torch.Tensor, optional
            The attention values of shape :math:`(E, *, H, 1)`. This is returned only
            when :attr:`get_attention` is ``True``.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )

            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, "fc_src"):
                    feat_src = self.fc(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats
                    )
                    feat_dst = self.fc(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats
                    )
                else:
                    feat_src = self.fc_src(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats
                    )
                    feat_dst = self.fc_dst(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats
                    )
            else:
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    *src_prefix_shape, self._num_heads, self._out_feats
                )
                if graph.is_block:
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
                    h_dst = h_dst[: graph.number_of_dst_nodes()]
                    dst_prefix_shape = (
                        graph.number_of_dst_nodes(),
                    ) + dst_prefix_shape[1:]

            # Linearly tranform the edge features.
            n_edges = edge_feat.shape[:-1]
            feat_edge = self.fc_edge(edge_feat).view(
                *n_edges, self._num_heads, self._out_feats
            )

            # Add edge features to graph.
            graph.edata["ft_edge"] = feat_edge

            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)

            # Calculate scalar for each edge.
            ee = (feat_edge * self.attn_edge).sum(dim=-1).unsqueeze(-1)
            graph.edata["ee"] = ee

            graph.srcdata.update({"ft": feat_src, "el": el})
            graph.dstdata.update({"er": er})
            # Compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v("el", "er", "e_tmp"))

            # e_tmp combines attention weights of source and destination node.
            # Add the attention weight of the edge.
            graph.edata["e"] = graph.edata["e_tmp"] + graph.edata["ee"]

            # Create new edges features that combine the
            # features of the source node and the edge features.
            graph.apply_edges(fn.u_add_e("ft", "ft_edge", "ft_combined"))

            e = self.leaky_relu(graph.edata.pop("e"))
            # Compute softmax.
            graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))

            # For each edge, element-wise multiply the combined features with
            # the attention coefficient.
            graph.edata["m_combined"] = (
                graph.edata["ft_combined"] * graph.edata["a"]
            )

            # First copy the edge features and then sum them up.
            graph.update_all(fn.copy_e("m_combined", "m"), fn.sum("m", "ft"))

            rst = graph.dstdata["ft"]
            # Residual.
            if self.res_fc is not None:
                # Use -1 rather than self._num_heads to handle broadcasting.
                resval = self.res_fc(h_dst).view(
                    *dst_prefix_shape, -1, self._out_feats
                )
                rst = rst + resval
            # Bias.
            if self.bias is not None:
                rst = rst + self.bias.view(
                    *((1,) * len(dst_prefix_shape)),
                    self._num_heads,
                    self._out_feats
                )
            # Activation.
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata["a"]
            else:
                return rst


class GATDotConv(nn.Module):
    r"""Graph attention layer from `Graph Attention Network
    <https://arxiv.org/pdf/1710.10903.pdf>`__

    .. math::
        h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{i,j} W^{(l)} h_j^{(l)}

    where :math:`\alpha_{ij}` is the attention score bewteen node :math:`i` and
    node :math:`j`:

    .. math::
        \alpha_{ij}^{l} &= \mathrm{softmax_i} (e_{ij}^{l})

        e_{ij}^{l} &= ({W_i^{(l)} h_i^{(l)}})^T \cdot {W_j^{(l)} h_j^{(l)}}
    """
    def __init__(self, in_feats, out_feats, num_heads, feat_drop=0.0, attn_drop=0.0, residual=False, activation=None, allow_zero_in_degree=False, bias=True):
        super(GATDotConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree

        # Linear transformations for source and destination nodes
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)

        # Dropout layers for feature and attention
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        # Residual connection
        self.has_linear_res = False
        self.has_explicit_bias = False
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=bias)
                self.has_linear_res = True
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer("res_fc", None)

        # Bias term (optional)
        if bias and not self.has_linear_res:
            self.bias = nn.Parameter(th.FloatTensor(size=(num_heads * out_feats,)))
            self.has_explicit_bias = True
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """
        Reinitialize learnable parameters.
        """
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        if self.has_explicit_bias:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
            if self.res_fc.bias is not None:
                nn.init.constant_(self.res_fc.bias, 0)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, edge_weight=None, get_attention=False):
        """
        Forward pass of the GATConv layer.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )

            # Feature dropout
            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, "fc_src"):
                    feat_src = self.fc(h_src).view(*src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc(h_dst).view(*dst_prefix_shape, self._num_heads, self._out_feats)
                else:
                    feat_src = self.fc_src(h_src).view(*src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(*dst_prefix_shape, self._num_heads, self._out_feats)
            else:
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(*src_prefix_shape, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_dst[:graph.number_of_dst_nodes()]
                    dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]

            # Compute attention scores using dot product
            graph.srcdata.update({"ft": feat_src})
            graph.dstdata.update({"ft": feat_dst})
            # 
            # e_{ij}^{(l)} = (W_i^{(l)} h_i^{(l)})^T \cdot (W_j^{(l)} h_j^{(l)})
            graph.apply_edges(fn.u_dot_v("ft", "ft", "e")) 
            # 
            e = graph.edata.pop("e") #
            graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))  # softmax  dropout

            if edge_weight is not None:
                graph.edata["a"] = graph.edata["a"] * edge_weight.tile(1, self._num_heads, 1).transpose(0, 2)

            # Message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]

            # Residual connection
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1, self._out_feats)
                rst = rst + resval

            # Bias term (optional)
            if self.has_explicit_bias:
                rst = rst + self.bias.view(*((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)

            # Activation function (optional)
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata["a"]
            else:
                return rst