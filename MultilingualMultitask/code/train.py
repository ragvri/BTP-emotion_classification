from model import Model
from setup import *
from generator import Generator
import torch
import torch.nn as nn
import fastText
import math
import numpy as np
from sklearn.metrics import *

class trainable_loss(nn.Module):
    def __init__(self):
        super(MY_LOSS, self).__init__()
        self.loss1 = nn.CrossEntropyLoss()
        self.loss2 = nn.MSELoss()
        self.w1 = torch.Tensor((1), requires_grad=True)
        self.w2 = torch.Tensor((1), requires_grad=True)

    def forward(self, pred1, tar1, pred2, tar2):

        loss1 = self.loss1(pred1, tar1)
        loss2 = self.loss2(pred2, tar2)

        combined_loss = loss1 * self.w1 + loss2*self.w2
        return combined_loss, loss1, loss2, self.w1, self.w2


def combined_loss(pred1, targ1, pred2, targ2):
    c_e = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    loss1 = c_e(pred1, targ1)
    # print(f'Loss1 : {loss1}')
    targ2 = targ2.float()
    loss2 = mse(pred2, targ2)
    return 0.5*loss1 + 0.5*loss2, loss1, loss2


gpu_id = '2'
train_file = f'./../dataset/joint_training_data/english/english_train_fold_0.csv'
test_file = f'./../dataset/joint_training_data/english/english_test_fold_0.csv'
emotions_dictionary = f'./../dataset/emoji_dictionary.json'
log_file = '/home1/zishan/raghav/emotion/MultilingualMultitask/logs/log.txt'
learning_rate = 1e-4
batch_size = 16
epochs = 200
seq_len = 75

f = open(log_file, 'w')
f.close()

setup_gpu(gpu_id)

ft = fastText.load_model('/home1/zishan/WordEmbeddings/FastText/wiki.en.bin')
labels2Idx = {'SADNESS': 0, 'FEAR/ANXIETY': 1, 'SYMPATHY/PENSIVENESS': 2, 'JOY': 3,
              'OPTIMISM': 4, 'NO-EMOTION': 5, 'DISGUST': 6, 'ANGER': 7, 'SURPRISE': 8}
train_generator = Generator(train_file, ft, labels2Idx, emotions_dictionary)
test_generator = Generator(test_file, ft, labels2Idx, emotions_dictionary)
model = Model(labels2Idx)
model = model.cuda()
optimiser = torch.optim.Adam(
    [i for i in model.parameters() if i.requires_grad], lr=learning_rate)

labels = [i for i, j in labels2Idx.items()]

# custom_loss = MY_LOSS()

best_f1 = 0
best_report = ""
best_epoch = 0
steps_per_epoch = math.ceil(train_generator.total_data()/batch_size)
for epoch in range(epochs):
    # Clear stored gradient
    print(f'Epoch: {epoch}/{epochs}')
    step = 0
    loss = 0
    optimiser.zero_grad()
    torch.set_grad_enabled(True)
    for x, y in train_generator.generate(batch_size, seq_len=seq_len):
        #if step % 50 == 0:
         #   print(step)
        if step == steps_per_epoch:
            break
        step = step + 1
        x_train, language = x
        x_train = torch.from_numpy(x_train).cuda()
        y_pred = model([x_train, language])
        y_pred_classification, y_pred_intensity = y_pred
        # y_pred_classification  = nn.Softmax(dim=-1)(y_pred_classification)
        y_train_classification, y_train_intensity = y
        y_train_classification = torch.from_numpy(y_train_classification).cuda()
        y_train_intensity = torch.from_numpy(y_train_intensity).cuda()

        loss, _, __ = combined_loss(y_pred_classification.view(-1, len(labels2Idx)),
                                    y_train_classification.view(-1), y_pred_intensity, y_train_intensity)
        loss.backward()  # gradients are stored when .backward is called
        optimiser.step()  # does the grad decent
        optimiser.zero_grad()  # clear away the gradients

    # Zero out gradient, else they will accumulate between epochs
    torch.set_grad_enabled(False)
    x_test, y_test = next(test_generator.generate(
        test_generator.total_data(), seq_len=seq_len))
    x_test, language = x_test
    y_test_classfication, y_test_intensity = y_test
    x_test = torch.from_numpy(x_test).cuda()

    test_pred = model([x_test, language])
    test_pred_classification, test_pred_intensity = test_pred

    y_test_classfication = torch.from_numpy(y_test_classfication).cuda()
    y_test_intensity = torch.from_numpy(y_test_intensity).cuda()

    #print(test_pred_classification.size())
    cb_loss, loss_classification, loss_intensity = combined_loss(test_pred_classification.view(
        -1, len(labels2Idx)), y_test_classfication.view(-1), test_pred_intensity, y_test_intensity)

    test_pred_classification = torch.nn.Softmax(
        dim=-1)(test_pred_classification)
    test_pred_classification = np.argmax(
        test_pred_classification.cpu().detach().numpy(), axis=-1)
    y_test_classfication = y_test_classfication.cpu().numpy()

    # print(x_test.shape,test_pred.shape)
    print("end epoch")

    c_r = classification_report(
        y_test_classfication, test_pred_classification, target_names=labels2Idx.keys())
    f1 = f1_score(y_test_classfication, test_pred_classification, average='macro')
    print(c_r)
    with open(log_file, 'a+') as f:
        to_write = f'Epoch:{epoch} Classification Loss: {loss_classification} Intensity loss: {loss_intensity} Accuracy:{accuracy_score(y_test_classfication , test_pred_classification)}\n{c_r}\n'
        f.write(to_write)

    if f1 > best_f1:
        best_f1 = f1
        best_report = c_r
        best_epoch = epoch
        torch.save(model.state_dict(), './../weights/weight.pth')

with open(log_file, 'a+') as f:
    to_write = f'Best classfication score at epoch {epoch}. Report:\n {best_report}'
    f.write(to_write)
