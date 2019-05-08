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
    return 0.7*loss1 + 0.3*loss2, loss1, loss2


gpu_id = '6'
train_file = f'./../dataset/joint_training_data/news_hindi_english_combined/train_fold_1.csv'
test_file = f'./../dataset/joint_training_data/news_hindi_english_combined/test_fold_1.csv'
emotions_dictionary = f'./../dataset/emoji_dictionary.json'
log_file = '/home1/zishan/raghav/emotion/MultilingualMultitask/logs/log_cross_lingual.txt'
learning_rate = 1e-4
batch_size = 1
epochs = 200
seq_len = 75

f = open(log_file, 'w')
f.close()

setup_gpu(gpu_id)

ft = fastText.load_model('/home1/zishan/WordEmbeddings/FastText/wiki.hi.bin')
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

best_f1_english = 0
best_f1_hindi = 0
best_report_english = ""
best_report_hindi = ""
best_epoch_hindi = 0
best_epoch_english = 0

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
    test_steps_per_epoch = math.ceil(test_generator.total_data()/batch_size)
    test_steps = 0
    y_test_classification_english = []
    y_test_classification_hindi = []
    y_test_intensity_english = []
    y_test_intensity_hindi = []
    test_pred_classification_hindi = []
    test_pred_classification_english = []
    test_pred_intensity_hindi = []
    test_pred_intensity_english = []

    for x_test, y_test in test_generator.generate(1, seq_len = seq_len):
        if test_steps == test_steps_per_epoch:
            break
        test_steps+=1
        x_test, language = x_test
        y_test_classfication, y_test_intensity = y_test
        

        x_test = torch.from_numpy(x_test).cuda()

        test_pred = model([x_test, language])
        test_pred_classification, test_pred_intensity = test_pred

        if language==1: # english
            y_test_classification_english.append(y_test_classfication)
            y_test_intensity_english.append(y_test_intensity)
            test_pred_classification_english.append(test_pred_classification)
            test_pred_intensity_english.append(test_pred_intensity)
        elif language==0:
            y_test_classification_hindi.append(y_test_classfication)
            y_test_intensity_hindi.append(y_test_hindi)
            test_pred_classification_hindi.append(test_pred_classification)
            test_pred_intensity_hindi.append(test_pred_intensity)

    y_test_classification_english = np.array(y_test_classification_english)
    y_test_intensity_english = np.array(y_test_intensity_english)
    y_test_classification_hindi = np.array(y_test_classification_hindi)
    y_test_intensity_hindi = np.array(y_test_intensity_hindi)

    y_test_classfication_english = torch.from_numpy(y_test_classfication_english).cuda()
    y_test_intensity_english = torch.from_numpy(y_test_intensity_english).cuda()

    y_test_classfication_hindi = torch.from_numpy(y_test_classfication_hindi).cuda()
    y_test_intensity_hindi = torch.from_numpy(y_test_intensity_hindi).cuda()

    test_pred_intensity_english = np.array(test_pred_intensity_english)
    test_pred_intensity_english = torch.from_numpy(test_pred_intensity_english).cuda()
    test_pred_classification_english = np.array(test_pred_classification_english)
    test_pred_classification_english = torch.from_numpy(test_pred_classification_english).cuda()

    test_pred_intensity_hindi = np.array(test_pred_intensity_hindi)
    test_pred_intensity_hindi = torch.from_numpy(test_pred_classification).cuda()
    test_pred_classification_hindi = np.array(test_pred_classification_hindi)
    test_pred_classification_hindi = torch.from_numpy(test_pred_classification_hindi).cuda()



    #print(test_pred_classification.size())
    cb_loss_english, loss_classification_english, loss_intensity_english = combined_loss(test_pred_classification_english.view(
        -1, len(labels2Idx)), y_test_classfication_english.view(-1), test_pred_intensity_english, y_test_intensity_english)

    
    cb_loss_hindi, loss_classification_hindi, loss_intensity_hindi = combined_loss(test_pred_classification_hindi.view(
        -1, len(labels2Idx)), y_test_classfication_hindi.view(-1), test_pred_intensity_hindi, y_test_intensity_hindi)

    test_pred_classification_english = torch.nn.Softmax(
        dim=-1)(test_pred_classification_english)
    test_pred_classification_english = np.argmax(
        test_pred_classification_english.cpu().detach().numpy(), axis=-1)
    y_test_classfication_english = y_test_classfication_english.cpu().numpy()

    
    test_pred_classification_hindi = torch.nn.Softmax(
        dim=-1)(test_pred_classification_hindi)
    test_pred_classification_hindi = np.argmax(
        test_pred_classification_hindi.cpu().detach().numpy(), axis=-1)
    y_test_classfication_hindi = y_test_classfication_hindi.cpu().numpy()

    # print(x_test.shape,test_pred.shape)
    print("end epoch")

    c_r_english = classification_report(
        y_test_classfication_english, test_pred_classification_english, target_names=labels2Idx.keys())
    f1_english = f1_score(y_test_classfication_english, test_pred_classification_english, average='macro')
    
    c_r_hindi = classification_report(
        y_test_classfication_hindi, test_pred_classification_hindi, target_names=labels2Idx.keys())
    f1_hindi = f1_score(y_test_classfication_hindi, test_pred_classification_hindi, average='macro')

    # print(c_r)
    with open(log_file, 'a+') as f:
        to_write = f'Epoch:{epoch} Classification Loss english: {loss_classification_english} Classification loss hindi: {loss_classification_hindi} Intensity loss english: {loss_intensity english} Intensity loss hindi: {loss_intensity hindi} Accuracy english:{accuracy_score(y_test_classfication_english , test_pred_classification_english) Accuracy english:{accuracy_score(y_test_classfication_hindi , test_pred_classification_hindi)}\n EN:\n{c_r_english}\n HI:\n{c_r_hindi}'
        f.write(to_write)

    if f1_english > best_f1_english:
        best_f1_english = f1_english
        best_report_english = c_r_english
        best_epoch_english = epoch
        torch.save(model.state_dict(), './../weights/weight_crosslingual_english.pth')

    if f1_hindi > best_f1_hindi:
        best_f1_hindi = f1_hindi
        best_report_hindi = c_r_hindi
        best_epoch_hindi = epoch
        torch.save(model.state_dict(), './../weights/weight_crosslingual_hindi.pth')


with open(log_file, 'a+') as f:
    to_write = f'Report_english:\n {best_report_english}\n Report hindi\n:{best_report_hindi}'
    f.write(to_write)
