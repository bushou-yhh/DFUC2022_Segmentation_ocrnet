_base_ = [
    '../_base_/models/ocrnet_hr18.py', '../_base_/datasets/dfuc.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
evaluation = dict(interval=800, metric='mDice', pre_eval=True)
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4)