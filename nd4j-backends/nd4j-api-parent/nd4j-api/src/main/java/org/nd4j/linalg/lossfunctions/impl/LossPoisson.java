package org.nd4j.linalg.lossfunctions.impl;

import lombok.EqualsAndHashCode;
import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Created by susaneraly on 9/9/16.
 */
@EqualsAndHashCode
public class LossPoisson implements ILossFunction {

    public INDArray scoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, INDArray exampleWeights) {
        /*
         mean of (yhat - y * log(yhat))
         */
        //INDArray postOutput = Nd4j.utioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFn, preOutput.dup()));
        INDArray postOutput = activationFn.getActivation(preOutput.dup(),true);

        INDArray scoreArr = Transforms.log(postOutput);
        scoreArr.muli(labels);
        scoreArr = postOutput.sub(scoreArr);

        if (exampleWeights != null) {
            if (exampleWeights.length() != scoreArr.size(0)) {
                throw new IllegalStateException("Example Weights vector (length " + exampleWeights.length() + ") does not match scoreArr.size(0)=" + scoreArr.size(0));
            }
            scoreArr.muliColumnVector(exampleWeights);
        }

        if (mask != null) scoreArr.muliColumnVector(mask);
        return scoreArr;
    }

    @Override
    public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, INDArray exampleWeights, boolean average) {
        INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask, exampleWeights);

        double score = scoreArr.sumNumber().doubleValue();

        if (average) score /= scoreArr.size(0);

        return score;
    }

    @Override
    public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, INDArray exampleWeights) {
        INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask, exampleWeights);
        return scoreArr.sum(1);
    }

    @Override
    public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, INDArray exampleWeights) {
        INDArray yHat = activationFn.getActivation(preOutput.dup(),true);
        INDArray yDivyhat = labels.div(yHat);
        INDArray dLda = yDivyhat.rsubi(1);

        if (exampleWeights != null) {
            if (exampleWeights.length() != dLda.size(0)) {
                throw new IllegalStateException("Example Weights vector (length " + exampleWeights.length() + ") does not match dLda.size(0)=" + dLda.size(0));
            }
            dLda.muliColumnVector(exampleWeights);
        }

        INDArray gradients = activationFn.backprop(preOutput, dLda).getFirst();     //TODO activation functions with params

        if (mask != null) {
            gradients.muliColumnVector(mask);
        }

        return gradients;
    }

    @Override
    public org.apache.commons.math3.util.Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, INDArray exampleWeights, boolean average) {
        //TODO: probably a more efficient way to do this...
        //Yes - will implement in round two. Just want to get done now.

        return new Pair<>(
                computeScore(labels, preOutput, activationFn, mask, exampleWeights, average),
                computeGradient(labels, preOutput, activationFn, mask, exampleWeights));
    }

    @Override
    public String toString() {
        return "LossPoisson()";
    }
}
