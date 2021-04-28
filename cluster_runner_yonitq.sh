#!/bin/bash
#SBATCH -J ML4J
#SBATCH -o logs/ML4J_%j.log
#SBATCH -N 4
#SBATCH --sockets-per-node=2
#SBATCH --cores-per-socket=18
#SBATCH --threads-per-core=1
#SBATCH --mem=250G
#SBATCH --mail-type=END
#SBATCH --mail-user=rotem.ovadia@mail.huji.ac.il
#SBATCH -A yonit-account
#SBATCH -p yonitq

source /opt/anaconda3/bin/activate ML4Jets
python benn.py
echo "Done"
