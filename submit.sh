#  training on drug data
for num in   1 2 3 4 5 
do
    python train_drug.py --order=1 --sample=4 --dropnode_rate=0.5 --seed $num
done

# training on simulation data
# for num in   1 2 3 4 5 
# do
#     python train_simulation.py --order=1 --sample=4 --dropnode_rate=0.5 --seed $num
# done



