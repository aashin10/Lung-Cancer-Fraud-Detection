# **🛡️ Lung Cancer Tampering Detection**
An AI-based system designed to detect fraudulent modifications in lung CT scans, specifically identifying artificially inserted cancerous nodules. This project leverages advanced image processing and machine learning to protect the integrity of medical imaging and support accurate lung cancer diagnosis.

## **📌 Overview**
Medical image tampering poses a serious threat to diagnosis, treatment, and patient trust. Our system addresses this by analyzing 3D lung CT scans to detect whether a nodule is real or fake. By integrating segmentation techniques and classification models, the system ensures that tampered cases are flagged effectively.

## **🎯 Objectives**                                       
Detect artificially inserted cancer nodules in CT scans.

Preserve the reliability of lung cancer diagnosis.

Enhance medical image security using AI.

Reduce false diagnoses and prevent healthcare fraud.

## **🧠 Core Workflow**                    
  
Preprocessing: Conversion of DICOM to Hounsfield Units (HU) and normalization.

Segmentation: Automatic extraction of lung regions and suspicious nodules.

Feature Extraction: Extraction of shape, texture, and intensity-based features.

Classification: Distinguish real vs. fake nodules using a machine learning model.

Visualization: Clearly display segmented nodules and prediction outcomes.

## **📈 Results**
Achieved an AUC-ROC score of 0.9452, indicating strong performance.

High accuracy in detecting fake nodules with minimal false positives.

Reliable model performance demonstrated through confusion matrix and feature importance analysis.

## **🔬 Dataset**                                          
Source: YMIRSKY Dataset

Format: 3D DICOM Lung CT Scans

Resolution: 512 × 512 × 512 voxels

## **🚀 Future Scope**                                        
Extend tampering detection to other cancers (breast, brain, liver).

Detect removal of real nodules to prevent false negatives.

Integrate recovery techniques to restore original scan integrity.

Apply the system in real-time clinical workflows.

## **📍 Relevance**                                        
This work contributes to secure AI in healthcare, ensuring that automated diagnostic systems remain trustworthy, transparent, and tamper-proof.
