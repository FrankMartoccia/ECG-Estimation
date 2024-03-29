function netPerformance = selectionCriterionActivity (FeaturesTraining, TargetTraining, FeaturesTesting, TargetTesting)
    net = patternnet(10, 'trainbr');
    
    net.divideParam.trainRatio = 1; 
    net.divideParam.valRatio = 0;
    net.divideParam.testRatio = 0;

    net.performFcn = 'mse';

    net.trainParam.showWindow = 0;
    
    net = train(net, FeaturesTraining', TargetTraining');
    
    predictedTarget = net(FeaturesTesting');
    netPerformance = perform(net, TargetTesting', predictedTarget);
end