#!/bin/bash
#SBATCH -J ML4J
#SBATCH -o logs/ML4J_%j.log
#SBATCH -N 4
#SBATCH -c 32
#SBATCH --mem=100G
#SBATCH --mail-type=END
#SBATCH --mail-user=rotem.ovadia@mail.huji.ac.il
#SBATCH -A yonit-account
#SBATCH -p yonitq

source /opt/anaconda3/bin/activate ML4Jets
python benn.py
echo "Done"
