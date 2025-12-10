import os
import sys
import time
import subprocess
import json
from datetime import datetime

# 配置
DATASETS = [
    "CICIoT2022",
    "USTC-TFC2016", 
    "ISCXTor2016",
    "ISCXVPN2016",
    "Bing-Traffic",
    "Malware-Traffic"
]

# 最优配置（基于之前的实验结果）
TEACHER_MODELS = ["tv_resnet152", "efficientnetv2_rw_m"]  # 双教师
STUDENT_MODELS = ["repvit_m0_9", "mobilenetv3_large_100", "efficientnet_b0"]  # Top 3学生
GPUS = [0, 1, 2]
BATCH_SIZE = 128
TEACHER_EPOCHS = 30  # 教师训练epoch
KD_EPOCHS = 50       # KD训练epoch

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")
    with open("training_pipeline.log", "a") as f:
        f.write(f"[{timestamp}] {msg}\n")

def run_cmd(cmd):
    log(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr

def check_gpu_status():
    ret, out, _ = run_cmd("nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits")
    if ret == 0:
        log(f"GPU Status:\n{out}")
    return out

def train_teachers(dataset, gpu_id):
    """训练教师模型"""
    log(f"=== Training Teachers for {dataset} on GPU {gpu_id} ===")
    teacher_paths = []
    
    for teacher in TEACHER_MODELS:
        log(f"Training teacher: {teacher}")
        cmd = f"""CUDA_VISIBLE_DEVICES={gpu_id} python training/train_teacher.py \
            --dataset {dataset} \
            --model {teacher} \
            --epochs {TEACHER_EPOCHS} \
            --batch_size {BATCH_SIZE} \
            --output checkpoints/{dataset}_teacher_{teacher}.pth"""
        
        ret, out, err = run_cmd(cmd)
        if ret == 0:
            teacher_path = f"checkpoints/{dataset}_teacher_{teacher}.pth"
            teacher_paths.append(teacher_path)
            log(f"✓ Teacher {teacher} trained successfully")
        else:
            log(f"✗ Teacher {teacher} training failed: {err}")
            
    return teacher_paths

def train_student_kd(dataset, student, teacher_paths, gpu_id):
    """使用KD训练学生模型"""
    log(f"=== Training Student {student} with KD for {dataset} on GPU {gpu_id} ===")
    
    teacher_arg = ",".join(teacher_paths)
    cmd = f"""CUDA_VISIBLE_DEVICES={gpu_id} python training/train_kd.py \
        --dataset {dataset} \
        --student {student} \
        --teachers {teacher_arg} \
        --epochs {KD_EPOCHS} \
        --batch_size {BATCH_SIZE} \
        --alpha 0.3 \
        --temperature 4.0 \
        --output checkpoints/{dataset}_student_{student}_kd.pth"""
    
    ret, out, err = run_cmd(cmd)
    if ret == 0:
        log(f"✓ Student {student} KD training completed")
        return f"checkpoints/{dataset}_student_{student}_kd.pth"
    else:
        log(f"✗ Student {student} KD training failed: {err}")
        return None

def evaluate_model(model_path, dataset, gpu_id):
    """评估模型"""
    log(f"=== Evaluating {model_path} on {dataset} ===")
    
    cmd = f"""CUDA_VISIBLE_DEVICES={gpu_id} python training/evaluate.py \
        --model_path {model_path} \
        --dataset {dataset} \
        --batch_size {BATCH_SIZE}"""
    
    ret, out, err = run_cmd(cmd)
    if ret == 0:
        log(f"✓ Evaluation completed:\n{out}")
        return out
    else:
        log(f"✗ Evaluation failed: {err}")
        return None

def update_experiment_log(dataset, student, results):
    """更新实验日志"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset,
        "student_model": student,
        "teacher_models": TEACHER_MODELS,
        "results": results
    }
    
    with open("EXPERIMENT_LOG.md", "a") as f:
        f.write(f"\n## {dataset} - {student} ({datetime.now().strftime('%Y-%m-%d %H:%M')})\n")
        f.write(f"- Teachers: {', '.join(TEACHER_MODELS)}\n")
        f.write(f"- Student: {student}\n")
        f.write(f"- Results: {results}\n")

def main():
    log("=" * 80)
    log("Starting Full KD Pipeline for All Datasets")
    log("=" * 80)
    
    check_gpu_status()
    
    # 为每个数据集运行完整流程
    for idx, dataset in enumerate(DATASETS):
        gpu_id = GPUS[idx % len(GPUS)]  # 轮流使用GPU
        
        log(f"\n{'='*80}")
        log(f"Processing Dataset: {dataset} (GPU {gpu_id})")
        log(f"{'='*80}\n")
        
        # Step 1: 训练教师模型
        teacher_paths = train_teachers(dataset, gpu_id)
        if not teacher_paths:
            log(f"✗ No teachers trained for {dataset}, skipping...")
            continue
        
        # Step 2: 为每个学生模型进行KD训练
        for student in STUDENT_MODELS:
            student_path = train_student_kd(dataset, student, teacher_paths, gpu_id)
            
            if student_path and os.path.exists(student_path):
                # Step 3: 评估学生模型
                results = evaluate_model(student_path, dataset, gpu_id)
                
                # Step 4: 记录结果
                update_experiment_log(dataset, student, results)
            
            # 检查GPU状态
            check_gpu_status()
            time.sleep(2)
        
        log(f"\n✓ Completed processing {dataset}\n")
    
    log("=" * 80)
    log("Full KD Pipeline Completed!")
    log("=" * 80)

if __name__ == "__main__":
    main()
