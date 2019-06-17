
import torch
from torch import nn
from Attention.utils.hyper import Hyperparams as hp
from Attention.modules import *

class RelationModuleMultiScale(torch.nn.Module):

    def __init__(self, img_feature_dim, num_frames, num_class):
        super(RelationModuleMultiScale, self).__init__()
        self.subsample_num = 3  # how many relations selected to sum up
        self.img_feature_dim = img_feature_dim
        self.scales = [i for i in range(num_frames, 1, -1)]  # generate the multiple frame relations

        self.relations_scales = []
        self.subsample_scales = []
        for scale in self.scales:
            relations_scale = self.return_relationset(num_frames, scale)
            self.relations_scales.append(relations_scale)
            self.subsample_scales.append(min(self.subsample_num, len(
                relations_scale)))  # how many samples of relation to select in each forward pass

        self.num_class = num_class
        self.num_frames = num_frames
        num_bottleneck = 256
        self.fc_fusion_scales = nn.ModuleList()  # high-tech modulelist
        self.transformer = AttModel(hp, 10000, 10000)
        self.encoder = EncoderAttModel(hp, 10000, 10000)

        self.final = nn.Sequential(
            nn.ReLU(),
            nn.Linear(num_frames * self.img_feature_dim, num_bottleneck),
            nn.ReLU(),
            nn.Linear(num_bottleneck, self.num_class),
        )

    def forward(self, input):
        # the first one is the largest scale
        act_all = input[:, self.relations_scales[0][0], :]
        act_all = self.encoder(act_all)

        high_level_feature = act_all.view((-1, act_all.size(1) * act_all.size(2)))
        high_level_result = self.final(high_level_feature)

        for scaleID in range(1, len(self.scales)):
            # iterate over the scales
            idx_relations_randomsample = np.random.choice(len(self.relations_scales[scaleID]),
                                                          self.subsample_scales[scaleID], replace=False)
            idx1, idx2, idx3 = idx_relations_randomsample
            act_relation1 = input[:, self.relations_scales[scaleID][idx1], :]
            act_relation2 = input[:, self.relations_scales[scaleID][idx2], :]
            act_relation3 = input[:, self.relations_scales[scaleID][idx3], :]


            temp_1 = self.transformer(act_relation1, act_all)
            temp_2 = self.transformer(act_relation2, act_all)
            temp_3 = self.transformer(act_relation3, act_all)

            act_all = act_all + temp_1 + temp_2 + temp_3

        act_feature = act_all.view((-1, act_all.size(1) * act_all.size(2)))
        act_all_result = self.final(act_feature)

        # type one: high_level
        # type two: low_level(random select one)
        # ada: only need first feature
        # tea: low_level(add all)



    def return_relationset(self, num_frames, num_frames_relation):
        import itertools
        return list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))

def return_TRN(img_feature_dim, num_frames, num_class):

    TRNmodel = RelationModuleMultiScale(img_feature_dim, num_frames, num_class)
    return TRNmodel



class AttModel(nn.Module):
    def __init__(self, hp_, enc_voc, dec_voc):

        super(AttModel, self).__init__()
        self.hp = hp_

        self.enc_voc = enc_voc
        self.dec_voc = dec_voc

        if self.hp.sinusoid:
            self.enc_positional_encoding = positional_encoding(num_units=self.hp.hidden_units,
                                                               zeros_pad=False,
                                                               scale=False)
        else:
            self.enc_positional_encoding = embedding(self.hp.maxlen, self.hp.hidden_units, zeros_pad=False, scale=False)
        self.enc_dropout = nn.Dropout(self.hp.dropout_rate)
        for i in range(self.hp.num_blocks):
            self.__setattr__('enc_self_attention_%d' % i, multihead_attention(num_units=self.hp.hidden_units,
                                                                              num_heads=self.hp.num_heads,
                                                                              dropout_rate=self.hp.dropout_rate,
                                                                              causality=False))
            self.__setattr__('enc_feed_forward_%d' % i, feedforward(self.hp.hidden_units,
                                                                    [4 * self.hp.hidden_units,
                                                                     self.hp.hidden_units]))

        if self.hp.sinusoid:
            self.dec_positional_encoding = positional_encoding(num_units=self.hp.hidden_units,
                                                               zeros_pad=False,
                                                               scale=False)
        else:
            self.dec_positional_encoding = embedding(self.hp.maxlen, self.hp.hidden_units, zeros_pad=False, scale=False)


        self.dec_dropout = nn.Dropout(self.hp.dropout_rate)
        for i in range(self.hp.num_blocks):
            self.__setattr__('dec_self_attention_%d' % i,
                             multihead_attention(num_units=self.hp.hidden_units,
                                                 num_heads=self.hp.num_heads,
                                                 dropout_rate=self.hp.dropout_rate,
                                                 causality=True))
            self.__setattr__('dec_vanilla_attention_%d' % i,
                             multihead_attention(num_units=self.hp.hidden_units,
                                                 num_heads=self.hp.num_heads,
                                                 dropout_rate=self.hp.dropout_rate,
                                                 causality=False))
            self.__setattr__('dec_feed_forward_%d' % i, feedforward(self.hp.hidden_units,
                                                                    [4 * self.hp.hidden_units,
                                                                     self.hp.hidden_units]))

        self.logits_layer = nn.Linear(self.hp.hidden_units, self.dec_voc)
        self.label_smoothing = label_smoothing()

    def forward(self, x, y):

        self.enc = x
        for i in range(self.hp.num_blocks):
            enc_temp = self.__getattr__('enc_self_attention_%d' % i)(self.enc, self.enc, self.enc)
            self.enc = self.enc + enc_temp
            self.enc = self.__getattr__('enc_feed_forward_%d' % i)(self.enc)

        self.dec = y

        for i in range(self.hp.num_blocks):

            # vanilla attention
            dec_temp = self.__getattr__('dec_vanilla_attention_%d' % i)(self.dec, self.enc, self.enc)
            self.dec = self.dec + dec_temp
            # feed forward
            self.dec = self.__getattr__('dec_feed_forward_%d' % i)(self.dec)

        return self.dec



class EncoderAttModel(nn.Module):
    def __init__(self, hp_, enc_voc, dec_voc):
 
        super(EncoderAttModel, self).__init__()
        self.hp = hp_

        self.enc_voc = enc_voc
        self.dec_voc = dec_voc


        if self.hp.sinusoid:
            self.enc_positional_encoding = positional_encoding(num_units=self.hp.hidden_units,
                                                               zeros_pad=False,
                                                               scale=False)
        else:
            self.enc_positional_encoding = embedding(self.hp.maxlen, self.hp.hidden_units, zeros_pad=False, scale=False)
        self.enc_dropout = nn.Dropout(self.hp.dropout_rate)
        for i in range(self.hp.num_blocks):
            self.__setattr__('enc_self_attention_%d' % i, multihead_attention(num_units=self.hp.hidden_units,
                                                                              num_heads=self.hp.num_heads,
                                                                              dropout_rate=self.hp.dropout_rate,
                                                                              causality=False))
            self.__setattr__('enc_feed_forward_%d' % i, feedforward(self.hp.hidden_units,
                                                                    [4 * self.hp.hidden_units,
                                                                     self.hp.hidden_units]))



    def forward(self, x):
        self.enc = x

        for i in range(self.hp.num_blocks):
            self.enc = self.__getattr__('enc_self_attention_%d' % i)(self.enc, self.enc, self.enc)
            self.enc = self.__getattr__('enc_feed_forward_%d' % i)(self.enc)

        return self.enc
