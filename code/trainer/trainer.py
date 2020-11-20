import os
import shutil
import time
import torch
from datetime import datetime

from data.process import load_json
from trainer.metric import evaluate
from trainer.util import build_spans


class Trainer():

    def __init__(self, args, device, model, optimizer,
                 scheduler, criterion, train_dataloader, val_dataloader,
                 ema, dev_eval_dict_path, identifier):
        self.args = args
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.dev_eval_dict = load_json(dev_eval_dict_path) # from tokens
        self.val_dataset = load_json(args.dev_data_filepath)
        self.ema = ema
        self.identifier = identifier

        self.start_time = datetime.now().strftime('%b-%d_%H-%M')
        self.start_epoch = 1
        self.step = 0
        self.best_em = 0
        self.best_f1 = 0
        if args.resume:
            self._resume_checkpoint(args.resume) # inplace resuming of all useful arguments
            self.model = self.model.to(self.device)
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

    def train(self):

        for epoch in range(self.start_epoch, self.args.epochs + 1):
            result = self._train_epoch(epoch)

            is_best = False
            if result["f1"] > self.best_f1:
                is_best = True
            if result["f1"] == self.best_f1 and result["em"] > self.best_em:
                is_best = True
            self.best_f1 = max(self.best_f1, result["f1"])
            self.best_em = max(self.best_em, result["em"])

            if epoch % self.args.save_freq == 0:
                self._save_checkpoint(
                    epoch, result["f1"], result["em"], is_best)

    def _train_epoch(self, epoch):
        self.model.train()
        self.model.to(self.device)

        # initialize
        global_loss = 0.0
        last_step = self.step - 1
        last_time = time.time()

        # train over batches
        for batch_idx, batch in enumerate(self.train_dataloader):
            # get batch
            (_,
             context_embeddings,
             question_embeddings,
             y1,
             y2,
             q_ids) = batch
            context_embeddings = context_embeddings.to(self.device)
            question_embeddings = question_embeddings.to(self.device)
            y1 = y1.to(self.device)
            y2 = y2.to(self.device)

            # calculate loss
            self.model.zero_grad()
            p1, p2 = self.model(context_embeddings, question_embeddings)

            loss1 = self.criterion(p1, y1)
            loss2 = self.criterion(p2, y2)
            # loss = torch.mean(loss1 + loss2) Errore
            loss = loss1 + loss2
            loss.backward()
            global_loss += loss.item()
            # gradient clip
            if self.args.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.grad_clip)

            # update model
            self.optimizer.step()

            # update learning rate
            if self.args.use_scheduler:
                self.scheduler.step()

            # exponential moving avarage
            if self.args.use_ema and self.ema is not None:
                self.ema(self.model, self.step)

            # print training info
            if self.step % self.args.print_freq == self.args.print_freq - 1:
                used_time = time.time() - last_time
                step_num = self.step - last_step
                speed = self.train_dataloader.batch_size * \
                        step_num / used_time
                batch_loss = global_loss / step_num
                print(("step: {}/{} \t "
                       "epoch: {} \t "
                       "lr: {} \t "
                       "loss: {} \t "
                       "speed: {} examples/sec").format(
                           batch_idx, len(self.train_dataloader),
                           epoch,
                           self.scheduler.get_lr(),
                           batch_loss,
                           speed))
                global_loss = 0.0
                last_step = self.step
                last_time = time.time()
            self.step += 1

            if self.args.debug and batch_idx >= self.args.debug_batchnum:
                break

        metrics = self._valid_epoch(self.dev_eval_dict, self.val_dataloader)
        print("dev_em: %f \t dev_f1: %f" % (
              metrics["exact_match"], metrics["f1"]))

        result = {}
        result["em"] = metrics["exact_match"]
        result["f1"] = metrics["f1"]
        return result

    def _valid_epoch(self, eval_dict, data_loader):
        """
        Evaluate model over development dataset.
        Return the metrics: em, f1.
        """
        if self.args.use_ema and self.ema is not None:
            self.ema.assign(self.model)

        self.model.eval()
        answer_dict = {}
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                # get batch
                (context_tokens,
                 context_embeddings,
                 question_embeddings,
                 y1,
                 y2,
                 q_ids) = batch
                context_embeddings = context_embeddings.to(self.device)
                question_embeddings = question_embeddings.to(self.device)
                y1 = y1.to(self.device)
                y2 = y2.to(self.device)

                p1, p2 = self.model(context_embeddings, question_embeddings)

                yp1 = torch.argmax(p1, 1)
                yp2 = torch.argmax(p2, 1)
                yps = torch.stack([yp1, yp2], dim=1)
                ymin, _ = torch.min(yps, 1)
                ymax, _ = torch.max(yps, 1)
                answer_dict_ = build_spans(context_tokens, q_ids, ymin.tolist(), ymax.tolist())
                answer_dict.update(answer_dict_)
                if((batch_idx + 1) == self.args.val_num_batches):
                    break

                if self.args.debug and batch_idx >= self.args.debug_batchnum:
                    break

        metrics = evaluate(self.dev_eval_dict, answer_dict)
        if self.args.use_ema and self.ema is not None:
            self.ema.resume(self.model)
        self.model.train()
        return metrics

    def _save_checkpoint(self, epoch, f1, em, is_best):
        if self.args.use_ema and self.ema is not None:
            self.ema.assign(self.model)
        arch = type(self.model).__name__
        state = {
            'epoch': epoch,
            'arch': arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_f1': self.best_f1,
            'best_em': self.best_em,
            'step': self.step + 1,
            'start_time': self.start_time}
        filename = os.path.join(
            self.args.save_dir,
            self.identifier +
            'checkpoint_epoch{:02d}_f1_{:.5f}_em_{:.5f}.pth.tar'.format(
                epoch, f1, em))
        print("Saving checkpoint: {} ...".format(filename))
        if not os.path.exists(self.args.save_dir):
            os.makedirs(self.args.save_dir)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(
                filename, os.path.join(self.args.save_dir, 'model_best.pth.tar'))
        if self.args.use_ema and self.ema is not None:
            self.ema.resume(self.model)
        return filename

    def _resume_checkpoint(self, resume_path):
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.best_f1 = checkpoint['best_f1']
        self.best_em = checkpoint['best_em']
        self.step = checkpoint['step']
        self.start_time = checkpoint['start_time']
        if self.args.use_scheduler:
            self.scheduler.last_epoch = checkpoint['epoch']
        print("Checkpoint '{}' (epoch {}) loaded".format(
            resume_path, self.start_epoch))