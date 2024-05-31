from . import evaluate
import utils

from torch.utils.data import ConcatDataset

#####################################################################
# callback for evaluating model-performance
def _eval_cb(log_interval, test_datasets, visdom=None, progress_dict=None, iters_per_task=None, test_size=None,
             classes_per_task=None, scenario="none", verbose=False):
    def eval_cb(classifier, batch, task=1, **kwargs):
        iteration = batch if task == 1 else (task-1) * iters_per_task + batch

        if iteration % log_interval == 0:
            evaluate.test_accuracy(classifier, test_datasets, task, iteration,
                                   classes_per_task=classes_per_task, scenario=scenario, progress_dict=progress_dict,
                                   test_size=test_size, visdom=visdom, verbose=verbose)

    return eval_cb if ((visdom is not None) or (progress_dict is not None)) else None


def _process_cb(log_interval, test_datasets, progress_dict, iters_per_task=None, test_size=None,
                classes_per_task=None, scenario="none", mid_size=None):
    assert progress_dict is not None
    def process_cb(classifier, batch, task=1, **kwargs):
        iteration = batch if task == 1 else (task - 1) * iters_per_task + batch
        # do the evaluation for the certain steps
        if iteration % log_interval == 0:
            print("starting record!")
            evaluate.check_neuron_activation(
                classifier, test_datasets, task, classes_per_task=classes_per_task, scenario=scenario,
                progress_dict=progress_dict, test_size=test_size, mid_size=mid_size,
            )

    return process_cb


#####################################################################
# callback for keeping track of training-progress
def _solver_loss_cb(log_interval, visdom=None, model=None, tasks=None, iters_per_task=None, epochs=None, rnt=None, replay=False,
                    progress_bar=True):

    def cb(bar, iter, loss_dict, task=1, epoch=None):
        iteration = iter if task == 1 else (task-1) * iters_per_task + iter

        #############################################
        # update the progress bar
        if progress_bar and bar is not None:
            task_stm = "" if (tasks is None) else f"Task: {task}/{tasks} |"
            epoch_stm = "" if ((epochs is None) or (epoch is None)) else f"Epochs: {epoch}/{epochs} |"
            bar.set_description(
                f'<MAIN MODEL> |{task_stm}{epoch_stm} training loss: {loss_dict["loss_total"]:.3f} | training accuracy: {loss_dict["accuracy"]:.3f}'
            )
            bar.update(1)

        #############################################
        # update the progress bar (to visdom, but i don't need it for now)
        if (iteration % log_interval == 0) and (visdom is not None):
            pass

    return cb
