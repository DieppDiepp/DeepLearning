import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import time
import logging
import numpy as np 
from sklearn.metrics import classification_report 
from torch.optim.lr_scheduler import StepLR 

# --- THÊM THƯ VIỆN ĐỂ ĐỌC THAM SỐ DÒNG LỆNH config ---
import argparse

from cMLP_1_layer import cMLP_1_layer
from cMLP_3_layers import cMLP_3_layers

# CÀI ĐẶT LOGGING
logger = logging.getLogger()
logger.setLevel(logging.INFO) 

if logger.hasHandlers():
    logger.handlers.clear()

# Tạo file log riêng biệt dựa trên tên mô hình sẽ được chọn sau
# Cấu hình file_handler bên trong hàm main
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
# --- KẾT THÚC CÀI ĐẶT LOGGING ---

# Cài đặt Hyperparameters
INPUT_SIZE = 28 * 28 
OUTPUT_SIZE = 10
LEARNING_RATE = 0.1
BATCH_SIZE = 64
NUM_EPOCHS = 15
LR_STEP_SIZE = 5 
LR_GAMMA = 0.1

# Chọn thiết bị (GPU nếu có để chạy kaggle)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Đang sử dụng thiết bị: {device}")

# Tải và chuẩn bị dữ liệu (MNIST)
# (Hàm này chỉ nên được gọi 1 lần)
def get_data_loaders(batch_size):
    logging.info("Đang tải dữ liệu MNIST...")
    train_data = datasets.MNIST(
        root="data", train=True, download=True, transform=ToTensor()
    )
    test_data = datasets.MNIST(
        root="data", train=False, download=True, transform=ToTensor()
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    logging.info("Tải dữ liệu hoàn tất.")
    return train_loader, test_loader

# Định nghĩa hàm Huấn luyện và Đánh giá
def train_and_evaluate(model_name, model, criterion, optimizer, scheduler, num_epochs, train_loader, test_loader): 
    logging.info(f"--- Bắt đầu huấn luyện: {model_name} ---")
    start_time = time.time()
    
    model.to(device)

    for epoch in range(num_epochs):
        model.train() 
        total_loss = 0
        current_lr = optimizer.param_groups[0]['lr']
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, INPUT_SIZE).to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels) 
            optimizer.zero_grad() 
            loss.backward()         
            optimizer.step()        
            total_loss += loss.item()
        
        logging.info(f'Epoch [{epoch+1}/{num_epochs}], LR: {current_lr:.5f}, Loss: {total_loss/len(train_loader):.4f}') 
        scheduler.step()

    training_time = time.time() - start_time
    logging.info(f"Huấn luyện xong trong {training_time:.2f} giây")

    # Vòng lặp Đánh giá (trên tập test)
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

# Hàm Main để xử lý logic chọn mô hình
def main(args):
    # Lấy dữ liệu
    train_loader, test_loader = get_data_loaders(BATCH_SIZE)
    
    model_choice = args.model
    model_name = ""
    model = None
    
    if model_choice == '1-layer':
        model_name = "MLP 1-Layer (SGD+Momentum)"
        model = cMLP_1_layer(INPUT_SIZE, OUTPUT_SIZE)
        
    elif model_choice == '3-layers':
        model_name = "MLP 3-Layers (SGD+Momentum)"
        model = cMLP_3_layers(INPUT_SIZE, OUTPUT_SIZE)
        
    else:
        # Trường hợp này không nên xảy ra nếu dùng 'choices' trong argparse
        logging.error(f"Lựa chọn mô hình không hợp lệ: {model_choice}")
        return

    # --- Cấu hình file log dựa trên lựa chọn ---
    log_filename = f'training_{model_choice}.log'
    file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8') 
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logging.info(f"--- Đã chọn mô hình: {model_name} ---")

    # Cài đặt optimizer và scheduler cho mô hình đã chọn
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9) 
    scheduler = StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    # Chạy huấn luyện và đánh giá
    train_and_evaluate(
        model_name=model_name, 
        model=model, 
        criterion=criterion, 
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=NUM_EPOCHS,
        train_loader=train_loader,
        test_loader=test_loader
    )

    logging.info(f"--- Hoàn tất huấn luyện {model_name} ---")
    # Xóa file handler để log không bị ghi đè nếu chạy lại
    logger.removeHandler(file_handler)
    file_handler.close()

# ĐIỂM BẮT ĐẦU CHẠY SCRIPT
if __name__ == "__main__":
    # 1. Tạo một parser
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình MLP trên MNIST.")
    
    # 2. Thêm tham số --model
    parser.add_argument(
        '--model',  # Tên của tham số
        type=str,
        required=True, # Bắt buộc người dùng phải cung cấp
        choices=['1-layer', '3-layers'], # Chỉ chấp nhận 2 giá trị này
        help="Chọn mô hình để huấn luyện: '1-layer' hoặc '3-layers'"
    )
    
    # 3. Đọc các tham số từ dòng lệnh
    args = parser.parse_args()
    
    # 4. Gọi hàm main với các tham số đã đọc
    main(args)