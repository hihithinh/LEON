# LEON Experiment Report - Index

## 📚 Tài liệu

### 1. [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md) ⭐ NEW
Tóm tắt kết quả chính với số liệu đã corrected

### 2. [QUICK_START.md](QUICK_START.md)
Hướng dẫn nhanh 3 bước để tạo plots

### 3. [PLOTS_ANALYSIS.md](../PLOTS_ANALYSIS.md)
Phân tích chi tiết từng plot với quan sát và kết luận

### 4. [EXPERIMENT_SUMMARY.md](EXPERIMENT_SUMMARY.md)
Tổng kết kết quả thực nghiệm với số liệu chi tiết

### 5. [README.md](README.md)
Hướng dẫn chi tiết đầy đủ về scripts và cách sử dụng

---

## 🎯 Kết quả chính (CORRECTED)

- ✅ **Speedup: 4.30x** vs PostgreSQL
- ✅ **Accuracy: 96.97%** average
- ✅ **Training: 60.41 hours** (2.5 days) ⭐ CORRECTED
- ✅ **Neural Network Training: 3.3 hours** (5.5% only!)
- ✅ **Data: 231,664 training pairs**
- ✅ **Models: 16** (level 2-17)

**Note**: Iteration 1 outlier (PowerShell pause 3 days) đã được loại bỏ

---

## 📊 6 Plots

1. **plot1_gmrl_performance.png** - GMRL convergence
2. **plot2_time_breakdown.png** - Time analysis
3. **plot3_data_growth.png** - Data accumulation
4. **plot4_model_performance.png** - Model metrics (2x2)
5. **plot5_convergence.png** - Convergence analysis
6. **plot6_learning_curves.png** - Learning curves

---

## 🚀 Quick Commands

```bash
# Extract data
python extract_training_data.py

# Generate plots
python plot_research_results.py
```

---

## 📝 Files

- `extract_training_data.py` - Data extraction script
- `plot_research_results.py` - Plotting script
- `plot*.png` - Generated plots (6 files)
