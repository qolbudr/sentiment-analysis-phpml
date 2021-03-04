<?php
require_once __DIR__ . '/vendor/autoload.php';

use Phpml\Dataset\ArrayDataset;
use Phpml\FeatureExtraction\TokenCountVectorizer;
use Phpml\Tokenization\WordTokenizer;
use Phpml\CrossValidation\RandomSplit;
use Phpml\Classification\DecisionTree;
use Phpml\Classification\NaiveBayes;
use Phpml\Metric\Accuracy;
use Phpml\Metric\ConfusionMatrix;
use Phpml\ModelManager;


$data = file_get_contents('sentiment.csv');
$arr = explode("\n", $data);
$label = [];
$sample = [];
for($i = 1; $i < count($arr);  $i++) {	
	$text = explode(",", $arr[$i]);
	$label[] = $text[1];
	$sample[] = $text[2];
}

$vectorizer = new TokenCountVectorizer(new WordTokenizer());

$vectorizer->fit($sample);
$vectorizer->transform($sample);

$dataset = new ArrayDataset($sample, $label);
$split_dataset = new RandomSplit($dataset);
$X_train = $split_dataset->getTrainSamples();
$y_train = $split_dataset->getTrainLabels();
$X_test  = $split_dataset->getTestSamples();
$y_test  = $split_dataset->getTestLabels();

$model = new NaiveBayes();
$model->train($X_train, $y_train);

$filepath = "model\\model.test";
$modelManager = new ModelManager();
$modelManager->saveToFile($model, $filepath);

$new = [
	"Kualitas film jelek susah dimengerti",
	"Menurut gw sejauh ini bagus banget film nya",
	"Bagus banget sumpah wajib nonton"
];

$vectorizer->transform($new);
$restoredClassifier = $modelManager->restoreFromFile($filepath);
$prediction = $restoredClassifier->predict($new);
print_r($prediction);