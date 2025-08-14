import os
import numpy as np
import torch
import torch.nn.functional as F

import lpips
from skimage.metrics import structural_similarity

from utils.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from utils.convert import pano_to_lidar
import numpy as np
import torch
import os
import numpy as np
import torch
from typing import List, Union
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from typing import List, Union

def fscore(dist1, dist2, threshold=0.001):
    """
    Calculates the F-score between two point clouds with the corresponding threshold value.
    :param dist1: Batch, N-Points
    :param dist2: Batch, N-Points
    :param th: float
    :return: fscore, precision, recall
    """
    # NB : In this depo, dist1 and dist2 are squared pointcloud euclidean
    # distances, so you should adapt the threshold accordingly.
    precision_1 = torch.mean((dist1 < threshold).float(), dim=1)
    precision_2 = torch.mean((dist2 < threshold).float(), dim=1)
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    fscore[torch.isnan(fscore)] = 0
    return fscore, precision_1, precision_2


class DepthMeter:
    def __init__(self, scale, lpips_fn=None):
        self.V = []
        self.N = 0
        self.scale = scale
        self.lpips_fn = lpips.LPIPS(net='alex').eval()

    def clear(self):
        self.V = []
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds = preds / self.scale
        truths = truths / self.scale
        preds, truths = self.prepare_inputs(
            preds, truths
        )  # [B, N, 3] or [B, H, W, 3], range[0, 1]

        # simplified since max_pixel_value is 1 here.
        depth_error = self.compute_depth_errors(truths, preds)

        depth_error = list(depth_error)
        self.V.append(depth_error)
        self.N += 1

    def compute_depth_errors(
        self, gt, pred, min_depth=1e-6, max_depth=80,
    ):  
        pred[pred < min_depth] = min_depth
        pred[pred > max_depth] = max_depth
        gt[gt < min_depth] = min_depth
        gt[gt > max_depth] = max_depth
        
        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        medae =  np.median(np.abs(gt - pred))

        lpips_loss = self.lpips_fn(torch.from_numpy(pred).squeeze(0), 
                                   torch.from_numpy(gt).squeeze(0), normalize=True).item()

        ssim = structural_similarity(
            pred.squeeze(0), gt.squeeze(0), data_range=np.max(gt) - np.min(gt)
        )

        psnr = 10 * np.log10(max_depth**2 / np.mean((pred - gt) ** 2))

        return rmse, medae, lpips_loss, ssim, psnr

    def measure(self):
        assert self.N == len(self.V)
        return np.array(self.V).mean(0)

    def write(self, writer, global_step, prefix="", suffix=""):
        writer.add_scalar(
            os.path.join(prefix, f"depth error{suffix}"), self.measure()[0], global_step
        )

    def report(self):
        return f"Depth_error = {self.measure()}"


class IntensityMeter:
    def __init__(self, scale, lpips_fn=None):
        self.V = []
        self.N = 0
        self.scale = scale
        self.lpips_fn = lpips.LPIPS(net='alex').eval()

    def clear(self):
        self.V = []
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds = preds / self.scale
        truths = truths / self.scale
        preds, truths = self.prepare_inputs(
            preds, truths
        )  # [B, N, 3] or [B, H, W, 3], range[0, 1]

        # simplified since max_pixel_value is 1 here.
        intensity_error = self.compute_intensity_errors(truths, preds)

        intensity_error = list(intensity_error)
        self.V.append(intensity_error)
        self.N += 1

    def compute_intensity_errors(
        self, gt, pred, min_intensity=1e-6, max_intensity=1.0,
    ):
        pred[pred < min_intensity] = min_intensity
        pred[pred > max_intensity] = max_intensity
        gt[gt < min_intensity] = min_intensity
        gt[gt > max_intensity] = max_intensity

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        medae =  np.median(np.abs(gt - pred))

        lpips_loss = self.lpips_fn(torch.from_numpy(pred).squeeze(0), 
                                   torch.from_numpy(gt).squeeze(0), normalize=True).item()

        ssim = structural_similarity(
            pred.squeeze(0), gt.squeeze(0), data_range=np.max(gt) - np.min(gt)
        )

        psnr = 10 * np.log10(max_intensity**2 / np.mean((pred - gt) ** 2))

        return rmse, medae, lpips_loss, ssim, psnr

    def measure(self):
        assert self.N == len(self.V)
        return np.array(self.V).mean(0)

    def write(self, writer, global_step, prefix="", suffix=""):
        writer.add_scalar(
            os.path.join(prefix, f"intensity error{suffix}"), self.measure()[0], global_step
        )

    def report(self):
        return f"Inten_error = {self.measure()}"


class RaydropMeter:
    def __init__(self, ratio=0.5):
        self.V = []
        self.N = 0
        self.ratio = ratio

    def clear(self):
        self.V = []
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(
            preds, truths
        )  # [B, N, 3] or [B, H, W, 3], range[0, 1]
        results = []

        rmse = (truths - preds) ** 2
        rmse = np.sqrt(rmse.mean())
        results.append(rmse)

        preds_mask = np.where(preds > self.ratio, 1, 0)
        acc = (preds_mask==truths).mean()
        results.append(acc)

        TP = np.sum((truths == 1) & (preds_mask == 1))
        FP = np.sum((truths == 0) & (preds_mask == 1))
        TN = np.sum((truths == 0) & (preds_mask == 0))
        FN = np.sum((truths == 1) & (preds_mask == 0))

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * (precision * recall) / (precision + recall)
        results.append(f1)

        self.V.append(results)
        self.N += 1

    def measure(self):
        assert self.N == len(self.V)
        return np.array(self.V).mean(0)

    def write(self, writer, global_step, prefix="", suffix=""):
        writer.add_scalar(os.path.join(prefix, "raydrop error"), self.measure()[0], global_step)

    def report(self):
        return f"Rdrop_error (RMSE, Acc, F1) = {self.measure()}"




class LabelMeter:
    def __init__(self):
        """
        Initializes the LabelMeter object for multi-class classification metrics.
        Handles softmax outputs and one-hot encoded ground truth labels.
        """
        self.V = []  # Store all metrics [accuracy, precision, recall, f1, rmse]
        self.N = 0   # Counter for number of updates

    def clear(self):
        """ Clears accumulated values. """
        self.V = []
        self.N = 0

    def prepare_inputs(self, *inputs: Union[np.ndarray, torch.Tensor]) -> List[np.ndarray]:
        """ 
        Prepares inputs (converts tensors to numpy arrays).
        Also converts softmax/one-hot to class indices if needed.
        """
        outputs = []
        for inp in inputs:
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            
            # Convert softmax/one-hot to class indices if needed
            if len(inp.shape) > 1 and inp.shape[-1] > 1:
                inp = np.argmax(inp, axis=-1)
                
            outputs.append(inp)
        return outputs

    def update(self, preds: Union[np.ndarray, torch.Tensor], truths: Union[np.ndarray, torch.Tensor]):
        """
        Updates accumulated metrics by comparing predictions to ground truth.
        
        Args:
            preds: Predicted probabilities (softmax output), shape [B, num_classes] or [B, H, W, num_classes]
            truths: Ground truth one-hot labels, shape [B, num_classes] or [B, H, W, num_classes]
        """
        preds, truths = self.prepare_inputs(preds, truths)
        
        # Flatten predictions and truths for metric calculation
        preds_flat = preds.flatten()
        truths_flat = truths.flatten()

        # Calculate accuracy
        accuracy = accuracy_score(truths_flat, preds_flat)

        # Calculate precision, recall, and f1 (weighted by support)
        precision, recall, f1, _ = precision_recall_fscore_support(
            truths_flat, 
            preds_flat, 
            average='weighted',
            zero_division=0
        )

        # # Calculate RMSE for probability predictions
        # rmse = np.sqrt(((truths_flat - preds_flat) ** 2).mean())

        # Calculate confusion matrix
        num_classes = len(np.unique(truths_flat))
        conf_matrix = confusion_matrix(
            truths_flat, 
            preds_flat, 
            labels=range(num_classes)
        )
        
        # # Calculate per-class accuracy (diagonal of normalized confusion matrix)
        per_class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
        mean_class_acc = np.mean(per_class_acc)

        # Store all metrics
        self.V.append([accuracy, mean_class_acc, precision, recall, f1])
        self.N += 1

    def measure(self):
        """ 
        Measures the average of all accumulated metrics.
        Returns: [accuracy, mean_class_accuracy, precision, recall, f1, rmse]
        """
        assert self.N == len(self.V), "Mismatch in accumulated metrics count."
        return np.array(self.V).mean(axis=0)

    def write(self, writer, global_step: int, prefix: str = "", suffix: str = ""):
        """ Writes metrics to TensorBoard. """
        metrics = self.measure()
        metric_names = ['accuracy', 'mean_class_accuracy', 'precision', 'recall', 'f1']
        
        for name, value in zip(metric_names, metrics):
            writer.add_scalar(
                os.path.join(prefix, f"label_{name}{suffix}"), 
                value, 
                global_step
            )

    def report(self):
        """ Reports the classification metrics. """
        acc, mca, prec, rec, f1 = self.measure()
        return (f"Label Metrics:\n"
                f"  Accuracy = {acc:.4f}\n"
                f"  Mean class acc. = {mca:.4f}\n"
                f"  Precision = {prec:.4f}\n"
                f"  Recall = {rec:.4f}\n"
                f"  F1 Score = {f1:.4f}\n")

class PointsMeter:
    def __init__(self, scale, intrinsics):
        self.V = []
        self.N = 0
        self.scale = scale
        self.intrinsics = intrinsics

    def clear(self):
        self.V = []
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds = preds / self.scale
        truths = truths / self.scale
        preds, truths = self.prepare_inputs(
            preds, truths
        )  # [B, N, 3] or [B, H, W, 3], range[0, 1]
        print("preds shape")
        print(preds.shape)
        
        chamLoss = chamfer_3DDist()
        pred_lidar = pano_to_lidar(preds[0], self.intrinsics)
        gt_lidar = pano_to_lidar(truths[0], self.intrinsics)
        print(pred_lidar[None, ...].shape)
        dist1, dist2, idx1, idx2 = chamLoss(
            torch.FloatTensor(pred_lidar[None, ...]).cuda(),
            torch.FloatTensor(gt_lidar[None, ...]).cuda(),
        )
        chamfer_dis = dist1.mean() + dist2.mean()
        threshold = 0.05  # monoSDF
        f_score, precision, recall = fscore(dist1, dist2, threshold)
        f_score = f_score.cpu()[0]

        self.V.append([chamfer_dis.cpu(), f_score])

        self.N += 1

    def measure(self):
        assert self.N == len(self.V)
        return np.array(self.V).mean(0)

    def write(self, writer, global_step, prefix="", suffix=""):
        writer.add_scalar(os.path.join(prefix, "CD"), self.measure()[0], global_step)

    def report(self):
        return f"Point_error (CD, F-score) = {self.measure()}"
    


