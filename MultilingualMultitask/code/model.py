import torch
import torch.nn as nn
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self, labels2Idx, input_dim=300, seq_len=75, output_dim=9,
                 num_layers=1):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = 50
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.loss = loss

        self.cnn2 = nn.Conv2d(
            in_channels=1, out_channels=self.hidden_dim, kernel_size=(2, self.hidden_dim*2))
        self.cnn3 = nn.Conv2d(1, self.hidden_dim, (3, self.hidden_dim*2))
        self.cnn4 = nn.Conv2d(1, self.hidden_dim, (4, self.hidden_dim*2))

        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim,
                            num_layers=self.num_layers, batch_first=True, bidirectional=True)
        # shared parameters over

        self.hindi_linear_classification = nn.Linear(
            self.seq_len*self.hidden_dim, 100)
        self.engilish_linear_classification = nn.Linear(
            self.seq_len*self.hidden_dim, 100)
        self.shared_linear_classification = nn.Linear(
            self.seq_len*self.hidden_dim, 100)

        self.hindi_linear_intensity = nn.Linear(
            self.seq_len*self.hidden_dim, 100)
        self.english_linear_intensity = nn.Linear(
            self.seq_len*self.hidden_dim, 100)
        self.shared_linear_intensity = nn.Linear(
            self.seq_len*self.hidden_dim, 100)

        self.linear_classification = nn.Linear(200, self.output_dim)
        self.linear_intensity = nn.Linear(200, 1)

        self.dropout = nn.Dropout(p=0.5)

        self.m2 = nn.ConstantPad2d((0, 0, 1, 0), 0)  # left, right, top, bottom
        self.m3 = nn.ConstantPad2d((0, 0, 1, 1), 0)
        self.m4 = nn.ConstantPad2d((0, 0, 2, 1), 0)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
            if type(m) in [nn.Conv2d, nn.Linear]:
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, ip, call=False):

        ip, language = ip[0], ip[1]

        lstm_out, self.hidden = self.lstm(ip)
        lstm_out = self.dropout(lstm_out)

        e2 = self.m2(lstm_out).unsqueeze(1)

        e3 = self.m3(lstm_out).unsqueeze(1)
        e4 = self.m4(lstm_out).unsqueeze(1)

        c2_out = self.cnn2(e2)
        c2_out = c2_out.squeeze(3)
        c2_out = F.relu(c2_out)
        c2_out = self.dropout(c2_out)
        c2_out = c2_out.contiguous().transpose(1, 2)

        c3_out = self.cnn3(e3)
        c3_out = c3_out.squeeze(3)
        c3_out = F.relu(c3_out)
        c3_out = self.dropout(c3_out)
        c3_out = c3_out.contiguous().transpose(1, 2)

        c4_out = self.cnn4(e4)
        c4_out = c4_out.squeeze(3)
        c4_out = F.relu(c4_out)
        c4_out = self.dropout(c4_out)
        c4_out = c4_out.contiguous().transpose(1, 2)

        sum_cnn_init = c2_out + c3_out + c4_out
        sum_cnn = sum_cnn_init.contiguous().view(-1, self.seq_len*self.hidden_dim) # contiguous: use whenever doing operations on tensor

        shared_cl_out = self.shared_linear_classification(sum_cnn)
        shared_in_out = self.shared_linear_intensity(sum_cnn)
        
        language = 1
        if language == 1:  # Hindi
            hi_cl_out = self.hindi_linear_classification(sum_cnn)
            hi_in_out = self.hindi_linear_intensity(sum_cnn)
            input_prefinal_cl = torch.cat((hi_cl_out, shared_cl_out), 1)
            input_prefinal_cl = input_prefinal_cl.contiguous()
            input_prefinal_in = torch.cat((hi_in_out,shared_in_out),1)
            input_prefinal_in = input_prefinal_in.contiguous()
        
        elif language == 0: # English
            en_cl_out = self.engilish_linear_classification(sum_cnn)
            en_in_out = self.english_linear_intensity(sum_cnn)
            input_prefinal_cl = torch.cat((en_cl_out,shared_cl_out),1)
            input_prefinal_cl = input_prefinal_cl.contiguous()
            input_prefinal_in = torch.cat((en_in_out,shared_in_out),1)
            input_prefinal_in = input_prefinal_in.contiguous()
        
        input_prefinal_cl = self.dropout(input_prefinal_cl)
        input_prefinal_in = self.dropout(input_prefinal_in)

        classification_pred = self.linear_classification(input_prefinal_cl)
        intensity_pred = self.linear_intensity(input_prefinal_in)

        return (classification_pred, intensity_pred)

if __name__ == "__main__":

    labels2Idx = {'SADNESS': 0, 'FEAR/ANXIETY': 1, 'SYMPATHY/PENSIVENESS': 2, 'JOY': 3,
                'OPTIMISM': 4, 'NO-EMOTION': 5, 'DISGUST': 6, 'ANGER': 7, 'SURPRISE': 8}
    gpu_id = '6'

    print(f'Using gpu {gpu_id}')
    setup_gpu(gpu_id)
    model = Model(labels2Idx=labels2Idx)


    inp = torch.Tensor(16, 75, 300)
    out = model(inp)
    print(f'out[0]: {out[0].size()} out[1]:{out[1].size()}')
