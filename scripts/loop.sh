for v in $(seq 1 10);
do
echo $v
sbatch scripts/generator.sh
sleep 2.1h
done
