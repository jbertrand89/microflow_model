import os
import torch
import numpy as np
from tqdm import tqdm
import time

from src.training_inference.evaluate import evaluate_training


def train(
        args, device, model, train_loader, val_loader, optimizer, scheduler=None, start_epoch=0, regression_loss_function=None, metrics=None):

    # Initialize metric arrays
    train_losses = np.zeros(args.epochs)
    val_losses = np.zeros(args.epochs)
    train_mae_error = np.zeros(args.epochs)
    val_mae_error = np.zeros(args.epochs)
    train_epe_error = np.zeros(args.epochs)
    val_epe_error = np.zeros(args.epochs)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None  # using mixed precision

    val_accuracy_mean_epoch_best = 10000

    t_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        # Get current learning rate
        epoch_lr = optimizer.param_groups[0]['lr']
        
        # Training step
        print(f'\n{"="*80}')
        print(f'TRAINING EPOCH {epoch + 1}/{args.epochs}  (Learning Rate: {epoch_lr:.6f})')
        print(f'{"="*80}\n')
        
        # todo check amp is correct
        train_losses[epoch], training_errors, training_errors_steps = train_one_epoch(
            args, device, model, train_loader, optimizer, scheduler, regression_loss_function,
            metrics, scaler, epoch_id=epoch)
        train_epe_error[epoch], train_mae_error[epoch] = training_errors
        
        # Print training metrics
        print("\nTraining Metrics:")
        print(f"  Loss:  {train_losses[epoch]:.4f}")
        print(f"  MAE:   {train_mae_error[epoch]:.4f}")
        print(f"  EPE:   {train_epe_error[epoch]:.4f}")

        # Validation step  
        print(f'\n{"="*80}')
        print(f'VALIDATION EPOCH {epoch + 1}/{args.epochs}')
        print(f'{"="*80}\n')
        
        val_losses[epoch], val_errors, val_errors_steps, val_errors_by_scaling_factor = evaluate_training(
            args, device, model, val_loader, regression_loss_function, metrics)
        val_epe_error[epoch], val_mae_error[epoch] = val_errors
        
        # Print validation metrics
        print("\nValidation Metrics:")
        print(f"  Loss:  {val_losses[epoch]:.4f}")
        print(f"  MAE:   {val_mae_error[epoch]:.4f}") 
        print(f"  EPE:   {val_epe_error[epoch]:.4f}")
        print(f"\n{'-'*80}")

        if args.save_offline:
            save_offline_metrics(
                args, epoch, train_losses, val_losses, train_mae_error, val_mae_error, train_epe_error, val_epe_error, epoch_lr,
                training_errors_steps, val_errors_steps, val_errors_by_scaling_factor, model.max_iterations)
            
        # Save checkpoint for recovery
        last_checkpoint_path = os.path.join(args.save_path, f"checkpoint_last_model.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, last_checkpoint_path)
        print(f"saved last {last_checkpoint_path}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.save_path, f"checkpoint_epoch{epoch + 1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_path)
            print(f"saved {checkpoint_path}")

        # Save the best checkpoint
        if val_mae_error[epoch] < val_accuracy_mean_epoch_best:
            val_accuracy_mean_epoch_best = val_mae_error[epoch]
            best_checkpoint_path = os.path.join(args.save_path, f"checkpoint_best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, best_checkpoint_path)

    t_end = time.time()
    print(f"total time {t_end - t_start}")


def train_one_epoch(
        args, device, model, train_loader, optimizer, scheduler, loss_function, metrics, scaler=None, epoch_id=0):

    model.train()

    epoch_train_loss= 0
    train_dataset_count = len(train_loader.dataset)  
    train_batches_count = train_dataset_count / args.train_batch_size  # this is actually the number of batches!
    training_errors = [0 for _ in metrics]
    training_errors_steps = [[0 for _ in metrics] for _ in range(model.max_iterations)]

    for _, batch in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        input = batch['pre_post_image'].to(device, non_blocking=args.non_blocking)
        target = batch['target_dm'].to(device, non_blocking=args.non_blocking)
        images_no_normalization = batch['pre_post_image_no_normalization'].to(device, non_blocking=args.non_blocking)
        b, _, h, w = input.shape
        input_model = torch.cat([input, images_no_normalization[:, 1].view(b, 1, h, w)], dim=1)

        with torch.cuda.amp.autocast(enabled=args.amp, dtype=torch.float16):
            optical_flow_predictions = model(input_model, args=args)

            # the loss can be computed for multi iterations or one iteration
            predicted = optical_flow_predictions if args.loss.lower().startswith("raft") else optical_flow_predictions[-1]
            regression_loss = loss_function(predicted, target)
            epoch_train_loss += regression_loss.item() / train_batches_count

            for metric_id, metric in enumerate(metrics):
                error = metric(optical_flow_predictions[-1], target)
                training_errors[metric_id] += error.item() / train_batches_count

                # metric at each step
                for i_flow in range(len(optical_flow_predictions)):
                    error = metric(optical_flow_predictions[i_flow], target)
                    training_errors_steps[i_flow][metric_id] += error.item() / train_batches_count

        if args.amp:
            scaler.scale(regression_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            regression_loss.backward()
            optimizer.step()

    scheduler.step()

    return epoch_train_loss, training_errors, training_errors_steps


def save_offline_metrics(
        args, epoch, train_losses, val_losses, train_mae_error, val_mae_error, train_epe_error, val_epe_error, epoch_lr, 
        training_errors_steps, val_errors_steps, val_errors_by_scaling_factor, max_iterations):
    train_step_errors = [str(training_errors_steps[step][1]) for step in range(max_iterations)]
    val_step_errors = [str(val_errors_steps[step][1]) for step in range(max_iterations)]
    scaling_factor_errors = [str(val_errors_by_scaling_factor[int(sf)][1]) for sf in args.val_scaling_factors]

    metrics_to_save = [
        str(epoch),
        str(train_losses[epoch]), 
        str(val_losses[epoch]),
        str(train_mae_error[epoch]), 
        str(val_mae_error[epoch]),
        str(train_epe_error[epoch]),
        str(val_epe_error[epoch]),
        str(epoch_lr)
    ]
    metrics_to_save.extend(scaling_factor_errors)
    metrics_to_save.extend(train_step_errors)
    metrics_to_save.extend(val_step_errors)

    # Save to file
    with open(args.save_offline_filename, "a+") as writer:
        writer.write(";".join(metrics_to_save) + "\n")
