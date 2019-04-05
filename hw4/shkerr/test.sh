echo 'boosted: digits 2 3'
./boosted_trees 2 3 data/digits_train.json data/digits_test.json > output/boosted_trees_2_3_digits.txt
echo 'boosted: heart 2 3'
./boosted_trees 2 3 data/heart_train.json data/heart_test.json > output/boosted_trees_2_3_heart.txt
echo 'boosted: mushrooms 2 3'
./boosted_trees 2 3 data/mushrooms_train.json data/mushrooms_test.json > output/boosted_trees_2_3_mushrooms.txt
echo 'boosted: winequality 2 3'
./boosted_trees 2 3 data/winequality_train.json data/winequality_test.json > output/boosted_trees_2_3_winequality.txt
echo 'boosted: digits 5 2'
./boosted_trees 5 2 data/digits_train.json data/digits_test.json > output/boosted_trees_5_2_digits.txt
echo 'boosted: heart 5 2'
./boosted_trees 5 2 data/heart_train.json data/heart_test.json > output/boosted_trees_5_2_heart.txt
echo 'boosted mushrooms 5 2'
./boosted_trees 5 2 data/mushrooms_train.json data/mushrooms_test.json > output/boosted_trees_5_2_mushrooms.txt
echo 'boosted winequality 5 2'
./boosted_trees 5 2 data/winequality_train.json data/winequality_test.json > output/boosted_trees_5_2_winequality.txt
echo 'bagged: digits 2 3'
./bagged_trees 2 3 data/digits_train.json data/digits_test.json > output/bagged_trees_2_3_digits.txt
echo 'bagged: heart 2 3'
./bagged_trees 2 3 data/heart_train.json data/heart_test.json > output/bagged_trees_2_3_heart.txt
echo 'bagged: mushrooms 2 3'
./bagged_trees 2 3 data/mushrooms_train.json data/mushrooms_test.json > output/bagged_trees_2_3_mushrooms.txt
echo 'bagged: winequality 2 3'
./bagged_trees 2 3 data/winequality_train.json data/winequality_test.json > output/bagged_trees_2_3_winequality.txt
echo 'bagged: digits 5 2'
./bagged_trees 5 2 data/digits_train.json data/digits_test.json > output/bagged_trees_5_2_digits.txt
echo 'bagged: heart 5 2'
./bagged_trees 5 2 data/heart_train.json data/heart_test.json > output/bagged_trees_5_2_heart.txt
echo 'bagged: mushrooms 5 2'
./bagged_trees 5 2 data/mushrooms_train.json data/mushrooms_test.json > output/bagged_trees_5_2_mushrooms.txt
echo 'bagged: winequality 5 2'
./bagged_trees 5 2 data/winequality_train.json data/winequality_test.json > output/bagged_trees_5_2_winequality.txt
echo 'cm: digits bag 5 2'
./confusion_matrix bag 5 2 data/digits_train.json data/digits_test.json > output/confusion_matrix_bag_5_2_digits.txt
echo 'cm: digits boost 5 2'
./confusion_matrix boost 5 2 data/digits_train.json data/digits_test.json > output/confusion_matrix_boost_5_2_digits.txt