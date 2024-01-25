import torch
import torch.nn as nn

class SegModel(nn.Module):

    def __init__(self, nLayers, inputChannels=3, outputChannels=1, lossWeights=None):
        super(SegModel, self).__init__()
        self.channelNum = 32
        self.resConvBlocksPerLayer = 4

        # First Convolution
        self.conv1 = nn.Conv2d(inputChannels, self.channelNum, (3, 3), padding=1)

        # NN Structure
        self.upsampleBlocks = []
        self.downsampleBlocks = []
        self.preLossConv = []
        for i in range(nLayers):
            self.upsampleBlocks.append(self.createRCBlock(self.resConvBlocksPerLayer))
            self.downsampleBlocks.append(self.createRCBlock(self.resConvBlocksPerLayer))
            self.preLossConv.append(nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(self.channelNum, outputChannels, 3, padding=1)
            ))

        self.upsampleBlocks.append(self.createRCBlock(self.resConvBlocksPerLayer))

        self.upsampleBlocks = nn.ModuleList(self.upsampleBlocks)
        self.downsampleBlocks = nn.ModuleList(self.downsampleBlocks)
        self.preLossConv = nn.ModuleList(self.preLossConv)

        self.losses = [nn.BCEWithLogitsLoss(posWeight=lw, reduction='none') for lw in lossWeights]
        self.losses = nn.ModuleList(self.losses)

