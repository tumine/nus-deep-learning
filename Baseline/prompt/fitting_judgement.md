结合 train_cnn_v2.py，解释在模型训练过程中如何判定模型的拟合状态。具体来说：
- 依靠什么指标判断模型发生过拟合？
- 依靠什么指标判断模型处在欠拟合状态？

要求：在 fitting_judgement.md 中追加写入中文纯文本，不使用图表、markdown 等富文本内容。

---

一、判断模型发生过拟合的指标

1. 训练准确率与验证准确率的差值（acc_gap）。若 acc_gap>0.10，认为存在过拟合；若 0.05 < acc_gap < 0.10，认为存在过拟合倾向。
2. 验证集总损失的变化趋势。若随着训练过程，val_loss 不再下降而反而开始上升，认为存在过拟合。

二、判断模型处在欠拟合状态的指标

1. 训练准确率（train_acc）持续偏低

2. 训练损失（train_loss）不收敛

3. 训练准确率与验证准确率同时偏低且差距很小

---

I. Indicators for Judging Overfitting

1. The gap between training accuracy and validation accuracy (acc_gap). If acc_gap exceeds 0.10, we consider the model as overfitting. If 0.05 < acc_gap < 0.10, we think there is a tendency of model overfitting.

2. The trend of validation loss (val_loss). If val_loss ceases to decrease and begins to rise as training progresses while training loss continues to fall, we think the model is overfit.

II. Indicators for Judging Underfitting

1. Both training accuracy and validation accuracy are low across epochs.

2. Training loss (train_loss) stays elevated and shows no clear downward trend over multiple epochs.
