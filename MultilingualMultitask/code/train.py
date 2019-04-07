from model import Model 
from setup import *
from generator import Generator
import torch
import torch.nn as nn


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
    loss1 = nn.CrossEntropyLoss(pred1, targ1)
    loss2 = nn.MSELoss(pred2, targ2)
    return 0.5*loss1 + 0.5*loss2, loss1, loss2

gpu_id = '6'
train_file = '/home1/zishan/raghav/emotion/MultilingualMultitask/dataset/train_file'
test_file = '/home1/zishan/raghav/emotion/MultilingualMultitask/dataset/test_file'
log_file = '/home1/zishan/raghav/emotion/MultilingualMultitask/logs/log.txt'
learning_rate = 1e-4
batch_size = 16
epochs = 100
seq_len = 75

setup_gpu(gpu_id)

train_generator = Generator(train_file)
model = Model()
optimiser = torch.optim.Adam([i for i in model.parameters() if i.requires_grad], lr=learning_rate)

labels2Idx = {'SADNESS': 0, 'FEAR/ANXIETY': 1, 'SYMPATHY/PENSIVENESS': 2, 'JOY': 3,
                'OPTIMISM': 4, 'NO-EMOTION': 5, 'DISGUST': 6, 'ANGER': 7, 'SURPRISE': 8}
labels = [i for i,j in labels2Idx.items()]

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
    for x,y in train_generator.generate(batch_size,seq_len=seq_len):
        if step == steps_per_epoch:
            break
        step = step + 1
        x_train, language = x
        x_train = torch.from_numpy(x).cuda()
        y_pred = model((x_train, language))
        y_pred_classification, y_pred_intensity = y_pred
        # y_pred_classification  = nn.Softmax(dim=-1)(y_pred_classification)
        y_train_classification, y_train_intensity = y
        y_train_classification = torch.from_numpy(y_train_classification).cuda()
        y_train_intensity = torch.from_numpy(y_train_intensity).cuda()

        loss, _, __ = combined_loss(y_pred_classification.view(-1,len(labels2Idx)), y_train_classification.view(-1), y_pred_intensity, y_train_intensity)
        loss.backward() # gradients are stored when .backward is called
        optimiser.step() # does the grad decent
        optimiser.zero_grad() # clear away the gradients

    # Zero out gradient, else they will accumulate between epochs
    torch.set_grad_enabled(False)
    x_test,y_test = next(test_generator.generate(test_generator.total_data(),seq_len=seq_len))
    x_test, language = x_test
    y_test_classfication, y_test_intensity = y_test 
    x_test = torch.from_numpy(x_test).cuda()

    test_pred = model((x_test), language)
    test_pred_classification, test_pred_intensity = test_pred

    test_pred_classification = torch.nn.Softmax(dim=-1)(test_pred_classification)
    test_pred_classification = np.argmax(test_pred.cpu().numpy(),axis=-1)

    combined_loss, loss_classification, loss_intensity = combined_loss(test_pred_classification.view(-1, len(labels2Idx)), y_test_classfication.view(-1), test_pred_intensity, y_test_intensity)

    # print(x_test.shape,test_pred.shape)
    print("end epoch")

    c_r = classification_report(y_test_classfication,test_pred_classification,target_names=labels2Idx.keys())
    f1 = f1_score(y_test,test_pred,average='macro')

    with open(log_file, 'a+') as f:
        to_write = f'Epoch:{epoch} Classification Loss: {loss_classification} Intensity loss: {loss_intensity} Accuracy:{accuracy_score(y_test_classfication , test_pred_classification)}\n{c_r}\n'
        f.write(to_write)

    if f1>best_f1:
        best_f1 = f1
        best_report = c_r
        best_epoch = epoch
        torch.save(model.state_dict(), 'weights/'+args.save_m)

with open(log_file, 'a+') as f:
    to_write = f'Best classfication score at epoch {epoch}. Report:\n {best_report}'