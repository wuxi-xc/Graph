from torch.nn.modules.module import Module
from functions.custom_func import JaggedLogSoftmax, JaggedArgmax, JaggedMax
# import networkx as nx

class JaggedLogSoftmaxModule(Module):
    def forward(self, logits, prefix_sum):
        return JaggedLogSoftmax()(logits, prefix_sum)

class JaggedArgmaxModule(Module):
    def forward(self, values, prefix_sum):
        return JaggedArgmax()(values, prefix_sum)

class JaggedMaxModule(Module):
    def forward(self, values, prefix_sum):
        return JaggedMax()(values, prefix_sum)

