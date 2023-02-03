python run.py --output_name main --base_model_name gpt2-xl --mask_filling_model_name t5-3b --n_perturbation_list 1,10,100 --n_samples 500 --pct_words_masked 0.3 --span_length 2
python run.py --output_name main --base_model_name EleutherAI/gpt-neo-2.7B --mask_filling_model_name t5-3b --n_perturbation_list 1,10,100 --n_samples 500 --pct_words_masked 0.3 --span_length 2
python run.py --output_name main --base_model_name EleutherAI/gpt-j-6B --mask_filling_model_name t5-3b --n_perturbation_list 1,10,100 --n_samples 500 --pct_words_masked 0.3 --span_length 2
python run.py --output_name main --base_model_name facebook/opt-2.7b --mask_filling_model_name t5-3b --n_perturbation_list 1,10,100 --n_samples 500 --pct_words_masked 0.3 --span_length 2
python run.py --output_name main --batch_size 20 --base_model_name EleutherAI/gpt-neox-20b --mask_filling_model_name t5-11b --n_perturbation_list 1,10,100 --n_samples 500 --pct_words_masked 0.3 --span_length 2

python run.py --output_name main_top_p --base_model_name gpt2-xl --mask_filling_model_name t5-3b --n_perturbation_list 1,10,100 --n_samples 500 --pct_words_masked 0.3 --span_length 2 --do_top_p
python run.py --output_name main_top_p --base_model_name EleutherAI/gpt-neo-2.7B --mask_filling_model_name t5-3b --n_perturbation_list 1,10,100 --n_samples 500 --pct_words_masked 0.3 --span_length 2 --do_top_p
python run.py --output_name main_top_p --base_model_name EleutherAI/gpt-j-6B --mask_filling_model_name t5-3b --n_perturbation_list 1,10,100 --n_samples 500 --pct_words_masked 0.3 --span_length 2 --do_top_p
python run.py --output_name main_top_p --base_model_name facebook/opt-2.7b --mask_filling_model_name t5-3b --n_perturbation_list 1,10,100 --n_samples 500 --pct_words_masked 0.3 --span_length 2 --do_top_p
python run.py --output_name main_top_p --batch_size 20 --base_model_name EleutherAI/gpt-neox-20b --mask_filling_model_name t5-11b --n_perturbation_list 1,10,100 --n_samples 500 --pct_words_masked 0.3 --span_length 2 --do_top_p

python run.py --output_name main_top_k --base_model_name gpt2-xl --mask_filling_model_name t5-3b --n_perturbation_list 1,10,100 --n_samples 500 --pct_words_masked 0.3 --span_length 2 --do_top_k
python run.py --output_name main_top_k --base_model_name EleutherAI/gpt-neo-2.7B --mask_filling_model_name t5-3b --n_perturbation_list 1,10,100 --n_samples 500 --pct_words_masked 0.3 --span_length 2 --do_top_k
python run.py --output_name main_top_k --base_model_name EleutherAI/gpt-j-6B --mask_filling_model_name t5-3b --n_perturbation_list 1,10,100 --n_samples 500 --pct_words_masked 0.3 --span_length 2 --do_top_k
python run.py --output_name main_top_k --base_model_name facebook/opt-2.7b --mask_filling_model_name t5-3b --n_perturbation_list 1,10,100 --n_samples 500 --pct_words_masked 0.3 --span_length 2 --do_top_k
python run.py --output_name main_top_k --batch_size 20 --base_model_name EleutherAI/gpt-neox-20b --mask_filling_model_name t5-11b --n_perturbation_list 1,10,100 --n_samples 500 --pct_words_masked 0.3 --span_length 2 --do_top_k



python run.py --output_name main --base_model_name gpt2-xl --mask_filling_model_name t5-3b --n_perturbation_list 1,10,100 --n_samples 312 --pct_words_masked 0.3 --span_length 2 --dataset squad --dataset_key context
python run.py --output_name main --base_model_name EleutherAI/gpt-neo-2.7B --mask_filling_model_name t5-3b --n_perturbation_list 1,10,100 --n_samples 312 --pct_words_masked 0.3 --span_length 2 --dataset squad --dataset_key context
python run.py --output_name main --base_model_name EleutherAI/gpt-j-6B --mask_filling_model_name t5-3b --n_perturbation_list 1,10,100 --n_samples 312 --pct_words_masked 0.3 --span_length 2 --dataset squad --dataset_key context
python run.py --output_name main --base_model_name facebook/opt-2.7b --mask_filling_model_name t5-3b --n_perturbation_list 1,10,100 --n_samples 312 --pct_words_masked 0.3 --span_length 2 --dataset squad --dataset_key context
python run.py --output_name main --batch_size 20 --base_model_name EleutherAI/gpt-neox-20b --mask_filling_model_name t5-11b --n_perturbation_list 1,10,100 --n_samples 312 --pct_words_masked 0.3 --span_length 2 --dataset squad --dataset_key context

python run.py --output_name main_top_p --base_model_name gpt2-xl --mask_filling_model_name t5-3b --n_perturbation_list 1,10,100 --n_samples 312 --pct_words_masked 0.3 --span_length 2 --do_top_p --dataset squad --dataset_key context
python run.py --output_name main_top_p --base_model_name EleutherAI/gpt-neo-2.7B --mask_filling_model_name t5-3b --n_perturbation_list 1,10,100 --n_samples 312 --pct_words_masked 0.3 --span_length 2 --do_top_p --dataset squad --dataset_key context
python run.py --output_name main_top_p --base_model_name EleutherAI/gpt-j-6B --mask_filling_model_name t5-3b --n_perturbation_list 1,10,100 --n_samples 312 --pct_words_masked 0.3 --span_length 2 --do_top_p --dataset squad --dataset_key context
python run.py --output_name main_top_p --base_model_name facebook/opt-2.7b --mask_filling_model_name t5-3b --n_perturbation_list 1,10,100 --n_samples 312 --pct_words_masked 0.3 --span_length 2 --do_top_p --dataset squad --dataset_key context
python run.py --output_name main_top_p --batch_size 20 --base_model_name EleutherAI/gpt-neox-20b --mask_filling_model_name t5-11b --n_perturbation_list 1,10,100 --n_samples 312 --pct_words_masked 0.3 --do_top_p --span_length 2 --dataset squad --dataset_key context

python run.py --output_name main_top_k --base_model_name gpt2-xl --mask_filling_model_name t5-3b --n_perturbation_list 1,10,100 --n_samples 312 --pct_words_masked 0.3 --span_length 2 --do_top_k --dataset squad --dataset_key context
python run.py --output_name main_top_k --base_model_name EleutherAI/gpt-neo-2.7B --mask_filling_model_name t5-3b --n_perturbation_list 1,10,100 --n_samples 312 --pct_words_masked 0.3 --span_length 2 --do_top_k --dataset squad --dataset_key context
python run.py --output_name main_top_k --base_model_name EleutherAI/gpt-j-6B --mask_filling_model_name t5-3b --n_perturbation_list 1,10,100 --n_samples 312 --pct_words_masked 0.3 --span_length 2 --do_top_k --dataset squad --dataset_key context
python run.py --output_name main_top_k --base_model_name facebook/opt-2.7b --mask_filling_model_name t5-3b --n_perturbation_list 1,10,100 --n_samples 312 --pct_words_masked 0.3 --span_length 2 --do_top_k --dataset squad --dataset_key context
python run.py --output_name main_top_k --batch_size 20 --base_model_name EleutherAI/gpt-neox-20b --mask_filling_model_name t5-11b --n_perturbation_list 1,10,100 --n_samples 312 --pct_words_masked 0.3 --do_top_k --span_length 2 --dataset squad --dataset_key context



python run.py --output_name main --base_model_name gpt2-xl --mask_filling_model_name t5-3b --n_perturbation_list 1,10,100 --n_samples 500 --pct_words_masked 0.3 --span_length 2 --dataset writing
python run.py --output_name main --base_model_name EleutherAI/gpt-neo-2.7B --mask_filling_model_name t5-3b --n_perturbation_list 1,10,100 --n_samples 500 --pct_words_masked 0.3 --span_length 2 --dataset writing
python run.py --output_name main --base_model_name EleutherAI/gpt-j-6B --mask_filling_model_name t5-3b --n_perturbation_list 1,10,100 --n_samples 500 --pct_words_masked 0.3 --span_length 2 --dataset writing
python run.py --output_name main --base_model_name facebook/opt-2.7b --mask_filling_model_name t5-3b --n_perturbation_list 1,10,100 --n_samples 500 --pct_words_masked 0.3 --span_length 2 --dataset writing
python run.py --output_name main --batch_size 20 --base_model_name EleutherAI/gpt-neox-20b --mask_filling_model_name t5-11b --n_perturbation_list 1,10,100 --n_samples 500 --pct_words_masked 0.3 --span_length 2 --dataset writing

python run.py --output_name main_top_p --base_model_name gpt2-xl --mask_filling_model_name t5-3b --n_perturbation_list 1,10,100 --n_samples 500 --pct_words_masked 0.3 --span_length 2 --do_top_p --dataset writing
python run.py --output_name main_top_p --base_model_name EleutherAI/gpt-neo-2.7B --mask_filling_model_name t5-3b --n_perturbation_list 1,10,100 --n_samples 500 --pct_words_masked 0.3 --span_length 2 --do_top_p --dataset writing
python run.py --output_name main_top_p --base_model_name EleutherAI/gpt-j-6B --mask_filling_model_name t5-3b --n_perturbation_list 1,10,100 --n_samples 500 --pct_words_masked 0.3 --span_length 2 --do_top_p --dataset writing
python run.py --output_name main_top_p --base_model_name facebook/opt-2.7b --mask_filling_model_name t5-3b --n_perturbation_list 1,10,100 --n_samples 500 --pct_words_masked 0.3 --span_length 2 --do_top_p --dataset writing
python run.py --output_name main_top_p --batch_size 20 --base_model_name EleutherAI/gpt-neox-20b --mask_filling_model_name t5-11b --n_perturbation_list 1,10,100 --n_samples 500 --pct_words_masked 0.3 --span_length 2 --do_top_p --dataset writing

python run.py --output_name main_top_k --base_model_name gpt2-xl --mask_filling_model_name t5-3b --n_perturbation_list 1,10,100 --n_samples 500 --pct_words_masked 0.3 --span_length 2 --do_top_k --dataset writing
python run.py --output_name main_top_k --base_model_name EleutherAI/gpt-neo-2.7B --mask_filling_model_name t5-3b --n_perturbation_list 1,10,100 --n_samples 500 --pct_words_masked 0.3 --span_length 2 --do_top_k --dataset writing
python run.py --output_name main_top_k --base_model_name EleutherAI/gpt-j-6B --mask_filling_model_name t5-3b --n_perturbation_list 1,10,100 --n_samples 500 --pct_words_masked 0.3 --span_length 2 --do_top_k --dataset writing
python run.py --output_name main_top_k --base_model_name facebook/opt-2.7b --mask_filling_model_name t5-3b --n_perturbation_list 1,10,100 --n_samples 500 --pct_words_masked 0.3 --span_length 2 --do_top_k --dataset writing
python run.py --output_name main_top_k --batch_size 20 --base_model_name EleutherAI/gpt-neox-20b --mask_filling_model_name t5-11b --n_perturbation_list 1,10,100 --n_samples 500 --pct_words_masked 0.3 --span_length 2 --do_top_k --dataset writing
