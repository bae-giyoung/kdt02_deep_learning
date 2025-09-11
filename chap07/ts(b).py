# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 한글 폰트 설정
plt.rcParams['font.family'] = ['Malgun Gothic']
plt.rcParams['axes.unicode_minus'] = False

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 디바이스: {device}")

# %%
def normalize_data(data):
    data_tensor = torch.FloatTensor(data) if not isinstance(data, torch.Tensor) else data
    data_min = torch.min(data_tensor)
    data_max = torch.max(data_tensor)
    normalized_data = (data_tensor - data_min) / (data_max - data_min)
    return normalized_data.numpy(), data_min.item(), data_max.item()

def denormalize_data(normalized_data, data_min, data_max):
    normalized_tensor = torch.FloatTensor(normalized_data) if not isinstance(normalized_data, torch.Tensor) else normalized_data
    denormalized = normalized_tensor * (data_max - data_min) + data_min
    return denormalized.numpy()

def calculate_mape(y_true, y_pred):
    y_true_tensor = torch.FloatTensor(y_true) if not isinstance(y_true, torch.Tensor) else y_true
    y_pred_tensor = torch.FloatTensor(y_pred) if not isinstance(y_pred, torch.Tensor) else y_pred
    mask = y_true_tensor != 0
    if mask.sum() == 0:
        return 0.0
    mape = torch.mean(torch.abs((y_true_tensor[mask] - y_pred_tensor[mask]) / y_true_tensor[mask])) * 100
    return mape.item()

def calculate_rmse(y_true, y_pred):
    y_true_tensor = torch.FloatTensor(y_true) if not isinstance(y_true, torch.Tensor) else y_true
    y_pred_tensor = torch.FloatTensor(y_pred) if not isinstance(y_pred, torch.Tensor) else y_pred
    mse = torch.mean((y_true_tensor - y_pred_tensor) ** 2)
    rmse = torch.sqrt(mse)
    return rmse.item()

def calculate_mae(y_true, y_pred):
    y_true_tensor = torch.FloatTensor(y_true) if not isinstance(y_true, torch.Tensor) else y_true
    y_pred_tensor = torch.FloatTensor(y_pred) if not isinstance(y_pred, torch.Tensor) else y_pred
    mae = torch.mean(torch.abs(y_true_tensor - y_pred_tensor))
    return mae.item()

# %%
# 1. 데이터셋 클래스
class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length]
        return torch.FloatTensor(x), torch.FloatTensor([y])

# %%
# 2. RNN 모델 정의
class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN 레이어
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# %%
# 3. LSTM 모델 정의 (확장용)
class SimpleLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# %%
# 4. 데이터 준비 함수 (PyTorch 기반)
def prepare_timeseries_data(data, column_name, sequence_length=20, train_ratio=0.8, val_ratio=0.1):
    print("=== 데이터 전처리 ===")
    
    if isinstance(data, pd.DataFrame):
        ts_data = data[column_name].values
        dates = data.index
    else:
        ts_data = data
        dates = None
    
    print(f"원본 데이터 크기: {len(ts_data)}")
    print(f"원본 데이터 범위: {ts_data.min():.2f} ~ {ts_data.max():.2f}")
    
    scaled_data, data_min, data_max = normalize_data(ts_data)
    print(f"정규화 후 범위: {scaled_data.min():.3f} ~ {scaled_data.max():.3f}")
    
    total_len = len(scaled_data)
    train_len = int(total_len * train_ratio)
    val_len = int(total_len * val_ratio)
    
    train_data = scaled_data[:train_len]
    val_data = scaled_data[train_len:train_len + val_len]
    test_data = scaled_data[train_len + val_len:]
    
    print(f"훈련 데이터: {len(train_data)}")
    print(f"검증 데이터: {len(val_data)}")
    print(f"테스트 데이터: {len(test_data)}")
    
    train_dataset = TimeSeriesDataset(train_data, sequence_length)
    val_dataset = TimeSeriesDataset(val_data, sequence_length)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    return {
        'original_data': ts_data,
        'dates': dates,
        'data_min': data_min,
        'data_max': data_max,
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'sequence_length': sequence_length
    }

# %%
# 5. 모델 생성 함수
def create_model(model_type='RNN', hidden_size=64, num_layers=2, dropout=0.2):
    print(f"=== {model_type} 모델 생성 ===")
    
    if model_type == 'RNN':
        model = SimpleRNN(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=1,
            dropout=dropout
        )
    elif model_type == 'LSTM':
        model = SimpleLSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=1,
            dropout=dropout
        )
    else:
        raise ValueError("model_type은 'RNN' 또는 'LSTM'이어야 합니다.")
    
    model = model.to(device)
    
    # 모델 정보 출력
    total_params = sum(p.numel() for p in model.parameters())
    print(f"총 파라미터 수: {total_params:,}")
    
    return model

# %%
# 6. 모델 훈련 함수
def train_model(model, data_dict, epochs=100, learning_rate=0.001, patience=15):
    print(f"=== 모델 훈련 시작 ===")
    
    train_loader = data_dict['train_loader']
    val_loader = data_dict['val_loader']
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # 훈련 모드
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.unsqueeze(-1).to(device)  # (batch, seq, 1)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # 검증 모드
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.unsqueeze(-1).to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        # 평균 손실 계산
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 최고 모델 저장
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}] - '
                  f'Train Loss: {avg_train_loss:.6f}, '
                  f'Val Loss: {avg_val_loss:.6f}, '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    # 최고 모델 로드
    model.load_state_dict(best_model_state)
    print(f"훈련 완료! 최고 검증 손실: {best_val_loss:.6f}")
    
    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }

# %%
# 7. 훈련 과정 시각화 함수
def plot_training_history(train_result):
    train_losses = train_result['train_losses']
    val_losses = train_result['val_losses']
    
    plt.figure(figsize=(10, 5))
    
    plt.plot(train_losses, label='훈련 손실', alpha=0.8)
    plt.plot(val_losses, label='검증 손실', alpha=0.8)
    plt.title('모델 훈련 과정')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"최종 훈련 손실: {train_losses[-1]:.6f}")
    print(f"최종 검증 손실: {val_losses[-1]:.6f}")

# %%
# 8. 예측 함수
def predict_sequence(model, input_seq):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).unsqueeze(-1).to(device)
        prediction = model(input_tensor)
        return prediction.cpu().numpy()[0, 0]

# %%
# 9. 미래 예측 함수
def forecast_future(model, data_dict, steps=10):
    print(f"\n=== {steps}일 예측 ===")
    
    test_data = data_dict['test_data']
    data_min = data_dict['data_min']
    data_max = data_dict['data_max']
    sequence_length = data_dict['sequence_length']
    
    # 마지막 시퀀스를 시작점으로 사용
    last_sequence = test_data[-sequence_length:].tolist()
    predictions = []
    
    for i in range(steps):
        # 예측
        pred = predict_sequence(model, last_sequence)
        predictions.append(pred)
        
        # 시퀀스 업데이트 (sliding window)
        last_sequence = last_sequence[1:] + [pred]
        
        if i < 5:  # 처음 5개만 출력
            # PyTorch 기반 역정규화
            pred_scaled = denormalize_data([pred], data_min, data_max)[0]
            print(f"{i+1}일 후: {pred_scaled:.2f}")
    
    # PyTorch 기반 역정규화
    predictions_scaled = denormalize_data(predictions, data_min, data_max)
    
    return predictions_scaled

# %%
# 10. 예측 결과 시각화 함수
def plot_forecast(data_dict, predictions, steps=10):
    original_data = data_dict['original_data']
    dates = data_dict['dates']
    
    plt.figure(figsize=(12, 6))
    
    # 최근 데이터 (마지막 50개)
    recent_len = min(50, len(original_data))
    recent_data = original_data[-recent_len:]
    
    if dates is not None:
        recent_dates = dates[-recent_len:]
        plt.plot(recent_dates, recent_data, label='실제 데이터', linewidth=2)
        
        # 미래 날짜 생성
        last_date = dates[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                   periods=steps, freq='D')
        plt.plot(future_dates, predictions, 
                label=f'{steps}일 예측', linewidth=2, color='red', linestyle='--')
    else:
        plt.plot(range(recent_len), recent_data, label='실제 데이터', linewidth=2)
        future_range = range(recent_len, recent_len + steps)
        plt.plot(future_range, predictions,
                label=f'{steps}일 예측', linewidth=2, color='red', linestyle='--')
    
    plt.title('RNN 시계열 예측 결과')
    plt.xlabel('시간')
    plt.ylabel('값')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# %%
# 11. 백테스트 함수 (PyTorch 버전)
def backtest_model(model, data_dict, test_steps=30):
    print(f"\n=== 백테스트 (최근 {test_steps}개) ===")
    
    test_data = data_dict['test_data']
    data_min = data_dict['data_min']
    data_max = data_dict['data_max']
    sequence_length = data_dict['sequence_length']
    
    if len(test_data) < test_steps:
        test_steps = len(test_data)
        print(f"테스트 데이터 부족으로 {test_steps}개로 조정")
    
    # 테스트 데이터에서 예측
    test_dataset = TimeSeriesDataset(test_data, sequence_length)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    predictions = []
    actuals = []
    
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            if len(predictions) >= test_steps:
                break
                
            batch_x = batch_x.unsqueeze(-1).to(device)
            output = model(batch_x)
            
            predictions.append(output.cpu().numpy()[0, 0])
            actuals.append(batch_y.numpy()[0, 0])
    
    # PyTorch 기반 역정규화
    predictions_scaled = denormalize_data(predictions, data_min, data_max)
    actuals_scaled = denormalize_data(actuals, data_min, data_max)
    
    # PyTorch 기반 성능 평가
    mape = calculate_mape(actuals_scaled, predictions_scaled)
    rmse = calculate_rmse(actuals_scaled, predictions_scaled)
    mae = calculate_mae(actuals_scaled, predictions_scaled)
    
    print(f"MAPE: {mape:.2f}%")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    
    # 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(actuals_scaled, label='실제값', linewidth=2)
    plt.plot(predictions_scaled, label='예측값', linewidth=2, alpha=0.8)
    plt.title(f'백테스트 결과 (MAPE: {mape:.2f}%, RMSE: {rmse:.2f})')
    plt.xlabel('시간')
    plt.ylabel('값')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return {
        'mape': mape,
        'rmse': rmse,
        'mae': mae,
        'predictions': predictions_scaled,
        'actuals': actuals_scaled
    }

# %%
# 12. 샘플 데이터 생성 함수
def create_sample_data():
    print("=== 샘플 데이터 생성 ===")
    periods = 1000
    np.random.seed(42)
    dates = pd.date_range('2002-07-01', periods=periods, freq='D')
    
    # 트렌드 + 노이즈가 있는 시계열
    trend = np.linspace(100, 150, periods)
    noise = np.random.normal(0, 5, periods)
    ts_data = trend + noise + np.sin(np.arange(periods) * 0.1) * 10
    
    sample_data = pd.DataFrame({
        'date': dates,
        'value': ts_data
    }).set_index('date')
    
    print(f"데이터 기간: {sample_data.index[0]} ~ {sample_data.index[-1]}")
    print(f"데이터 크기: {len(sample_data)}")
    print(f"값 범위: {sample_data['value'].min():.2f} ~ {sample_data['value'].max():.2f}")
    
    return sample_data

# %%
# 13. 전체 실행 함수 (수업용 통합 예제)
def run_complete_example(model_type='RNN'):
    print(f"=== {model_type} 완전한 예제 실행 ===\n")
    
    # 1. 데이터 생성
    sample_data = create_sample_data()
    
    # 2. 데이터 전처리
    data_dict = prepare_timeseries_data(sample_data, 'value', sequence_length=20)
    
    # 3. 모델 생성
    model = create_model(model_type=model_type, hidden_size=64, num_layers=2)
    
    # 4. 모델 훈련
    train_result = train_model(model, data_dict, epochs=50, learning_rate=0.001)
    
    # 5. 훈련 과정 시각화
    plot_training_history(train_result)
    
    # 6. 미래 예측
    predictions = forecast_future(train_result['model'], data_dict, steps=10)
    
    # 7. 예측 시각화
    plot_forecast(data_dict, predictions)
    
    # 8. 백테스트
    backtest_result = backtest_model(train_result['model'], data_dict)
    
    return {
        'data_dict': data_dict,
        'model': train_result['model'],
        'train_result': train_result,
        'predictions': predictions,
        'backtest_result': backtest_result
    }

# %%
# 실행 예제
if __name__ == "__main__":
    # 개별 함수 실행 예제
    print("=== 개별 함수 실행 예제 ===")

    # 데이터 생성
    sample_data = create_sample_data()

    # 원본 데이터 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(sample_data.index, sample_data['value'])
    plt.title('원본 시계열 데이터')
    plt.xlabel('날짜')
    plt.ylabel('값')
    plt.grid(True, alpha=0.3)
    plt.show()

    data_dict = prepare_timeseries_data(sample_data, 'value')
    model = create_model(model_type='RNN')
    train_result = train_model(model, data_dict, epochs=50)
    plot_training_history(train_result)
    predictions = forecast_future(train_result['model'], data_dict, steps=10)
    plot_forecast(data_dict, predictions)
    backtest_result = backtest_model(train_result['model'], data_dict)
# %%
