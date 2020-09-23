#!/bin/bash
#SBATCH -J ML4J
#SBATCH -o logs/ML4J_%j.log
#SBATCH -N 2
#SBATCH -c 20
#SBATCH --mem=40G
#SBATCH --mail-type=END
#SBATCH --mail-user=rotem.ovadia@mail.huji.ac.il
#SBATCH -A rotemov-account
#SBTACH -p yonitq,allq

source /opt/anaconda3/bin/activate ML4Jets
python benn.py
echo "Done"
