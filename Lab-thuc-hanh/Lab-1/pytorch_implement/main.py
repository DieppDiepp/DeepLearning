import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import time
import logging
import numpy as np 
from sklearn.metrics import classification_report 
# --- THÊM THƯ VIỆN SCHEDULER ---
from torch.optim.lr_scheduler import StepLR 
# -------------------------------

# Import 2 class model của bạn
from cMLP_1_layer import cMLP_1_layer
from cMLP_3_layers import cMLP_3_layers

# --- 1. CÀI ĐẶT LOGGING ---
logger = logging.getLogger()
logger.setLevel(logging.INFO) 

if logger.hasHandlers():
    logger.handlers.clear()

file_handler = logging.FileHandler('training_log.log', mode='w', encoding='utf-8') 
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)
# --- KẾT THÚC CÀI ĐẶT LOGGING ---


# --- 2. Cài đặt Hyperparameters ---
INPUT_SIZE = 28 * 28 
OUTPUT_SIZE = 10
LEARNING_RATE = 0.1 # <-- TĂNG LR BAN ĐẦU CHO SGD + MOMENTUM
BATCH_SIZE = 100
NUM_EPOCHS = 15
# --- THAM SỐ CHO LR SCHEDULER ---
LR_STEP_SIZE = 5 # Giảm LR sau mỗi 5 epochs
LR_GAMMA = 0.1   # Giảm LR xuống 10% (0.1 lần)
# --------------------------------

# --- 3. Chọn thiết bị (GPU nếu có) ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Đang sử dụng thiết bị: {device}")

# --- 4. Tải và chuẩn bị dữ liệu (MNIST) ---
train_data = datasets.MNIST(
    root="data", train=True, download=True, transform=ToTensor()
)
test_data = datasets.MNIST(
    root="data", train=False, download=True, transform=ToTensor()
)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# --- 5. Định nghĩa hàm Huấn luyện và Đánh giá ---

# --- THAY ĐỔI: Thêm scheduler vào tham số ---
def train_and_evaluate(model_name, model, criterion, optimizer, scheduler, num_epochs): 
    logging.info(f"--- Bắt đầu huấn luyện: {model_name} ---")
    start_time = time.time()
    
    model.to(device)

    # Vòng lặp huấn luyện
    for epoch in range(num_epochs):
        model.train() 
        total_loss = 0
        # --- Lấy LR hiện tại để log ---
        current_lr = optimizer.param_groups[0]['lr']
        # ----------------------------
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, INPUT_SIZE).to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels) 
            optimizer.zero_grad() 
            loss.backward()       
            optimizer.step()      
            total_loss += loss.item()
        
        # --- THAY ĐỔI: Log cả LR ---
        logging.info(f'Epoch [{epoch+1}/{num_epochs}], LR: {current_lr:.5f}, Loss: {total_loss/len(train_loader):.4f}') 
        # ----------------------------

        # --- THAY ĐỔI: Cập nhật scheduler sau mỗi epoch ---
        scheduler.step()
        # ------------------------------------------------

    training_time = time.time() - start_time
    logging.info(f"Huấn luyện xong trong {training_time:.2f} giây")

    # --- 6. Vòng lặp Đánh giá (trên tập test) ---
    model.eval() 
    all_labels = []
    all_predictions = []
    with torch.no_grad(): 
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, INPUT_SIZE).to(device)
            labels = labels.to(device)
            outputs = model(images) 
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy()) 
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    logging.info(f'Độ chính xác TỔNG THỂ (Accuracy) của {model_name}: {accuracy:.2f} %')
    logging.info(f"--- ĐÁNH GIÁ CHI TIẾT TỪNG CHỮ SỐ (0-9) CHO {model_name} ---")
    report_str = classification_report(all_labels, all_predictions, digits=4)
    for line in report_str.split('\n'):
        if line.strip(): 
             logging.info(line)
    
    return accuracy, training_time

# --- 7. Chạy mô hình 1-layer ---
model_1 = cMLP_1_layer(INPUT_SIZE, OUTPUT_SIZE)
criterion_1 = nn.CrossEntropyLoss()
# --- THAY ĐỔI: Dùng SGD với momentum ---
optimizer_1 = torch.optim.SGD(model_1.parameters(), lr=LEARNING_RATE, momentum=0.9) 
# --- THAY ĐỔI: Tạo scheduler ---
scheduler_1 = StepLR(optimizer_1, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

train_and_evaluate(
    model_name="MLP 1-Layer (SGD+Momentum)", # <-- Đổi tên cho rõ
    model=model_1, 
    criterion=criterion_1, 
    optimizer=optimizer_1,
    scheduler=scheduler_1, # <-- Truyền scheduler vào
    num_epochs=NUM_EPOCHS
)

# --- 8. Chạy mô hình 3-layers ---
model_3 = cMLP_3_layers(INPUT_SIZE, OUTPUT_SIZE)
criterion_3 = nn.CrossEntropyLoss()
# --- THAY ĐỔI: Dùng SGD với momentum ---
optimizer_3 = torch.optim.SGD(model_3.parameters(), lr=LEARNING_RATE, momentum=0.9)
# --- THAY ĐỔI: Tạo scheduler ---
scheduler_3 = StepLR(optimizer_3, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

train_and_evaluate(
    model_name="MLP 3-Layers (SGD+Momentum)", # <-- Đổi tên cho rõ
    model=model_3,
    criterion=criterion_3,
    optimizer=optimizer_3,
    scheduler=scheduler_3, # <-- Truyền scheduler vào
    num_epochs=NUM_EPOCHS
)

logging.info("--- Hoàn tất ---")