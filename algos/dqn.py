import os, random
import torch
import torch.nn.functional as F
from torch.optim import Adam
from core.utils import soft_update, hard_update

# 須同時修改 erl_trainer.py, 梯度裁剪(?)
class DQN(object):
    def __init__(self, args, model_constructor):

        self.gamma = args.gamma
        self.tau = args.tau
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = model_constructor.make_model('CategoricalPolicy').to(device=self.device)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.actor_lr)

        self.actor_target = model_constructor.make_model('CategoricalPolicy').to(device=self.device)
        hard_update(self.actor_target, self.actor)

        self.num_updates = 0
        #self.gtM = 0. #20220619 最大梯度絕對值
        #self.lossAcc = [] #20220622 loss趨勢

    def update_parameters(self, state_batch, next_state_batch, action_batch, reward_batch, done_batch):

        state_batch = state_batch.to(self.device)
        next_state_batch=next_state_batch.to(self.device)
        action_batch=action_batch.to(self.device)
        reward_batch=reward_batch.to(self.device)
        done_batch=done_batch.to(self.device)

        action_batch = action_batch.long().unsqueeze(1)
        with torch.no_grad():
            _, _, ns_logits = self.actor_target.noisy_action(next_state_batch, return_only_action=False)
            labels_next = ns_logits.detach().max(1)[0].unsqueeze(1)
            next_q_value = reward_batch + (1-done_batch) * self.gamma * labels_next

        _, _, logits  = self.actor.noisy_action(state_batch, return_only_action=False)
        q_val = logits.gather(1, action_batch)

        q_loss = (next_q_value - q_val)**2
        q_loss = q_loss.mean()
        #self.lossAcc.append(q_loss.item()/state_batch.shape[0]) #20220622 loss趨勢
        #print('LLLLLL  L55 q_loss/batch_size: ',q_loss.item()/state_batch.shape[0])#20220621 q_loss/每epoch(與batch_size無關)
        #該值還必須傳送出去才能顯示趨勢; batch_size由--batchsize設定
        self.actor_optim.zero_grad()
        q_loss.backward()
        #print('LLLLLL  L59 max/min grad: ',torch.max((self.actor.adv.weight.grad))) #adv.weight.grad=> -177 ~ 324(目前觀測值)      
        """
        ##20220623 梯度裁剪       
        for param in self.actor.parameters():
          param.grad.data.clamp_(-1, 1)

        torch.nn.utils.clip_grad_norm_(parameters=self.actor.parameters(), max_norm=10)#20220628 对一组可迭代(网络)参数的梯度范数进行裁剪。https://blog.csdn.net/Mikeyboi/article/details/119522689        
        ##20220619 最大梯度絕對值
        if torch.max(torch.abs((self.actor.adv.weight.grad))) > self.gtM:
          self.gtM = torch.max(torch.abs((self.actor.adv.weight.grad)))
        elif torch.max(torch.abs((self.actor.adv.bias.grad))) > self.gtM:
          self.gtM = torch.max(torch.abs((self.actor.adv.bias.grad)))
        """
        self.actor_optim.step()

        self.num_updates += 1
        soft_update(self.actor_target, self.actor, self.tau)

