Multiscale Convolutional Recurrent Neural Network and Data Augmentation For Polyphonic Audio Event Classification based on DCASE2019 task3
 references:
Sharath Adavanne and Archontis Politis and Tuomas£¬A multi-room reverberant dataset for sound event localization and detection.

Documentation See http://dcase.community/challenge2019/task-sound-event-localization-and-detection-results

Getting started
Clone repository from Github or download latest release.
Install requirements with command: pip install -r requirements.txt
Run the application with default settings: python .py
System description
This is the Multiscale Convolutional Recurrent Neural Network and Data Augmentation For Polyphonic Audio Event Classification For Sound Event Detection for the Detection and Classification of Acoustic Scenes and Events 2019 (DCASE2019) challenge task 3.

We present Multiscale Convolutional Recurrent Neural Networks(MCRNN) and Data Augmentation method for polyphonic audio event classification which consists of audio feature extraction, data augmentation, and audio event classifier. We extract the log Mel spectrum as input spectrum image features. We propose a different Background Spectrum Random Replacement(BSRR) data augmentation method, which uses Standard Normal Distribution Data instead of the original time-domain, frequency domain or time-frequency domain background spectrum features with randomly selected position and length. The audio event classifier constitutes of Multiscale Convolutional  Neural Networks(MCNN), Recurrent Neural Networks(RNN) and the multiclass output. The proposed method is tested on the IEEE Detection and Classification of Acoustic Scenes and Events (DCASE2019) challenge dataset. The experimental results showed that the BSRR data augmentation method can be efficient in improving audio event classification performance. We achieved the best results by combining our BSRR and other different data augmentation method. The performance outperforms the challenge baseline, improving F1 from  0.799 to 0.939$\pm$0.002 and reducing error rate from 0.34 to 0.11$\pm$0.01 on the development dataset. On the evaluation dataset, our method achieved an error rate(ER) of 0.06$\pm$0.01 and 0.967$\pm$0.008 of F1 score, which got an improvement of 79${\%}$ and 13${\%}$ than the baseline. The F1 and ER is better than that of on the development dataset, exhibits good generalization capabilities of the systems.

The main approach implemented in the system:

Acoustic features: Log Mel-band energies extracted in 40ms windows with 20ms hop size.
.
©À©¤©¤ seld_sed_dev.py          # train on the development dataset.
©À©¤©¤ seld_sed_eval.py         # test on the evaluation dataset.
©¦   ©¸©¤©¤ parameter.py          # parameters
©À©¤©¤ RM_dev.py         # Background Spectrum Random Replacement(BSRR) Data Augmentation method  on the development dataset.
©À©¤©¤ RM_eval.py         # Background Spectrum Random Replacement(BSRR) Data Augmentation method   on the evaluation dataset.
©À©¤©¤ pytorch_model.py         # models.
©À©¤©¤ README.md               # This file
©¸©¤©¤ requirements.txt        # External module dependencies 
Installation
The system is developed for Python 3.6.5. Thesystem is tested to work with Linux operating systems.

See more detailed instructions from documentation.
references:
Multiscale Convolutional Recurrent Neural Network and Data Augmentation For Polyphonic Audio Event Classification
License
The DCASE Framework is released only for academic research under EULA.pdf from Tampere University of Technology.