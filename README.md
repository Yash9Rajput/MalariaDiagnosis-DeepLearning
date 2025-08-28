🧬 Malaria Diagnosis using Deep Learning

Malaria is a life-threatening disease caused by parasites, and early and accurate diagnosis is critical. This project leverages Deep Learning with TensorFlow & Keras to build an automated system for classifying cell images as Parasitized or Uninfected.

The goal is to build, evaluate, and improve deep learning models using modern AI practices such as data augmentation, callbacks, hyperparameter tuning, and visualization tools to achieve high diagnostic accuracy.

📌 Features

✔️ Data preparation, cleaning & preprocessing

✔️ Exploratory data analysis & visualization

✔️ Image augmentation for generalization

✔️ Model building using TensorFlow & Keras

✔️ Evaluation with classification metrics & ROC plots

✔️ TensorFlow Callbacks for performance improvement:

Early Stopping

Learning Rate Scheduling

Model Checkpoints

CSV Logger
✔️ TensorBoard integration for live monitoring

✔️ Hyperparameter Tuning for optimization

✔️ Comparison of different models & architectures

✔️ Final testing on unseen dataset

🛠 Tech Stack & Libraries

Deep Learning Frameworks: TensorFlow, Keras

Data Analysis & Visualization: NumPy, Pandas, Matplotlib, Seaborn

Model Evaluation: Scikit-learn (classification reports, ROC, confusion matrix)

Performance Monitoring: TensorBoard, TensorFlow Callbacks

Hyperparameter Optimization: Keras Tuner

📊 Project Workflow

Data Preparation & Preprocessing

Load dataset (cell images)

Normalize & resize images

Train-validation-test split

Exploratory Data Analysis (EDA)

Class distribution

Visualization of infected vs healthy cells

Data Augmentation

Rotation, zoom, flip, brightness adjustments

Model Development

Baseline CNN model

Advanced deep learning models

Evaluation & Metrics

Accuracy, Precision, Recall, F1-score

ROC & AUC analysis

Performance Improvements

Callbacks (EarlyStopping, LR Scheduler, CSV Logger, ModelCheckpoint)

Hyperparameter tuning with Keras Tuner

Integration with TensorBoard

Testing & Final Results

Evaluate on unseen data

Compare model variants

📂 Repository Structure
├── Malarial_Diagnosis.ipynb   # Main notebook (data prep, visualization, modeling)

├── Advanced_Models.ipynb      # Advanced CNN models & tuning

├── data/                      # Dataset folder (link provided separately)

├── outputs/                   # Model checkpoints, logs, ROC plots

├── README.md                  # Project documentation

🚀 How to Run

Clone the repository:

git clone https://github.com/your-username/malaria-diagnosis.git
cd malaria-diagnosis


Install dependencies:

pip install -r requirements.txt


Run the Jupyter notebooks:

jupyter notebook


(Optional) Launch TensorBoard for monitoring:

tensorboard --logdir=logs/

📈 Results

Achieved high classification accuracy on test set

ROC-AUC values close to 1.0

Improved model performance with callbacks & hyperparameter tuning

(Include accuracy/ROC curve plots here – add images in /outputs and link them in README)

🔮 Future Work

Experiment with Transfer Learning (e.g., VGG16, ResNet, EfficientNet)

Deployment as a Web App (Streamlit / Flask)

Integration with Mobile Devices for real-world diagnostic support

📸 Demo


[View](https://drive.google.com/file/d/13MfHGUtYPsqbh_1JzEG0GlJnVEvCugHk/view?usp=sharing)

🤝 Contributing

Contributions are welcome! Feel free to fork this repo, create a branch, and submit a PR.

📜 License

This project is licensed under the MIT License.