

diffu_values=(-0.5)
sigma_values=(0.001)  
seeds=(1)

for diffu in "${diffu_values[@]}"; do
    for seed in "${seeds[@]}"; do
        for sigma in "${sigma_values[@]}"; do
            echo "Running with diffu=$diffu and sigma=$sigma"
            python main.py --problem-name single_cell \
                        --sde-type tbm \
                        --forward-net res \
                        --backward-net res \
                        --num-res-block 3 \
                        --dir single_cell/sb \
                        --gpu 0 \
                        --diffu $diffu \
                        --sigma $sigma \
                        --seed $seed
        done
    done
done
