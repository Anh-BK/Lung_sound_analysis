#!/bin/sh
echo '=============='
grep COPD patient_diagnosis.csv |wc -l
grep Asthma patient_diagnosis.csv |wc -l
grep Bronchiectasis patient_diagnosis.csv |wc -l

echo '=============='
grep URTI patient_diagnosis.csv |wc -l
grep LRTI patient_diagnosis.csv |wc -l
grep Pneumonia patient_diagnosis.csv |wc -l
grep Bronchiolitis patient_diagnosis.csv |wc -l

echo '=============='
grep Healthy patient_diagnosis.csv |wc -l
