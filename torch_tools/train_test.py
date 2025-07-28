import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import roc_curve, auc
from torchvision.ops.focal_loss import sigmoid_focal_loss
import torch.nn.functional as F
import numpy as np


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def train(net,ckp_path,train_loader,validate_loader,optimizer,n_epochs,verbose = False,use_best_valid = True,use_f1= True):
    net.to(device)
    # 损失函数:这里用交叉熵
    loss_function = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,4,0.7)

    best_result = 0.0
    # 开始进行训练和验证，训练一轮，验证一轮
    for epoch in range(n_epochs):
        # train
        net.train()    #训练过程中，使用之前定义网络中的dropout
        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
            images, labels = data
            # print(labels)
            outputs = net(images.to(device))
            # print(outputs)
            labels_one_hot = F.one_hot(labels, num_classes=outputs.shape[1])
            # loss = sigmoid_focal_loss(outputs, labels_one_hot.float().to(device),reduction='mean',gamma=0.5,alpha=0.5)
            loss = loss_function(outputs, labels_one_hot.float().to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # print train process
            rate = (step + 1) / len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
        scheduler.step()
        
        if np.isnan(running_loss):
            break
        # valid
        net.eval()    #测试过程中不需要dropout，使用所有的神经元
        with torch.no_grad():
            predict_y = []
            label_y = []
            for step, data in enumerate(validate_loader, start=0):
                test_images, test_label = data
                outputs = net(test_images.to(device))
                predict_y.append(torch.max(outputs, dim=1)[1].cpu())
                label_y.append(test_label.cpu())
            predict_y = torch.cat(predict_y).numpy()
            label_y = torch.cat(label_y).numpy()
            precision = precision_score(label_y, predict_y, average='micro',zero_division=0.0)
            recall = recall_score(label_y, predict_y, average='micro',zero_division=0.0)
            f1= f1_score(label_y, predict_y, average='micro',zero_division=0.0)
            acc = accuracy_score(label_y, predict_y)
            if use_f1:
                result = f1
            else:
                result = acc
        print('[epoch %d] train_loss: %f  val_result: %.2f' % (epoch + 1, running_loss/len(train_loader),result),end="")
        if verbose:
            print(f' precision:{precision:.2f},recall:{recall:.2f},f1:{f1:.2f},acc:{acc} ',end="")
        if result >= best_result:
            best_result = result
        # save
        if result >= best_result or not use_best_valid:
            torch.save(net.state_dict(), ckp_path)
    # return acc
    if use_best_valid :
        return best_result
    else:
        return result

def test(net,ckp_path,test_loader,verbose = True):
    net.load_state_dict(torch.load(ckp_path))
    net.to(device)
    # test
    net.eval()    #测试过程中不需要dropout，使用所有的神经元
    with torch.no_grad():
        predict_y = []
        label_y = []
        for step, data in enumerate(test_loader, start=0):
            test_images, test_label = data
            outputs = net(test_images.to(device))
            predict_y.append(torch.max(outputs, dim=1)[1].cpu())
            label_y.append(test_label.cpu())
        predict_y = torch.cat(predict_y).numpy()
        label_y = torch.cat(label_y).numpy()
        predict_y =  [1 - label for label in predict_y]
        label_y =  [1 - label for label in label_y]

        precision = precision_score(label_y, predict_y)
        recall = recall_score(label_y, predict_y)
        f1= f1_score(label_y, predict_y)
        acc = accuracy_score(label_y, predict_y)
    if verbose:
        print(f'precision:{precision},recall:{recall},f1:{f1},acc:{acc}')
    return {"precision":precision,"recall":recall,"f1":f1,"acc":acc}


def test_roc(net,ckp_path,test_loader):
    mean_fpr = np.linspace(0, 1, 100)
    net.load_state_dict(torch.load(ckp_path))
    net.to(device)
    # test
    net.eval()    #测试过程中不需要dropout，使用所有的神经元
    with torch.no_grad():
        predict_y = []
        label_y = []
        for step, data in enumerate(test_loader, start=0):
            test_images, test_label = data
            outputs = net(test_images.to(device))
            predict_y.append(outputs[:,1].cpu())
            label_y.append(test_label.cpu())
        predict_y = torch.cat(predict_y).numpy()
        label_y = torch.cat(label_y).numpy()
        fpr, tpr, thresholds = roc_curve(label_y, predict_y)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        roc_auc = auc(fpr, tpr)
    return mean_fpr,interp_tpr,roc_auc