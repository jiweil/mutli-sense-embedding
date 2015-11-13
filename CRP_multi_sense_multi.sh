javac CRP_multi_sense.java
java CRP_multi_sense -dimension 300 -word_window 15 -save_file this_CRP -frequency_file ../../Wiki_Data/release/frequency.txt -train_file ../../Wiki_Data/release/train_file1.txt -load_embedding 1 -embedding_file ../../Wiki_Data/release/all/small_vect
#java CRP_multi_sense_multi1 -dimension 300 -word_window 15 -save_file CRP -frequency_file ../../Wiki_Data/release/frequency.txt -train_file ../../Wiki_Data/release/train_file1.txt -load_embedding 1 -embedding_file ../../Wiki_Data/release/all/small_vect
