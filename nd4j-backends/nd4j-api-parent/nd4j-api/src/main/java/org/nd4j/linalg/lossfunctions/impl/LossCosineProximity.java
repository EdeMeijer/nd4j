package org.nd4j.linalg.lossfunctions.impl;

import lombok.EqualsAndHashCode;
import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;

/**
 * Created by susaneraly on 9/9/16.
 */
@EqualsAndHashCode
public class LossCosineProximity implements ILossFunction {

    public INDArray scoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, INDArray exampleWeights) {
        /*
         mean of -(y.dot(yhat)/||y||*||yhat||)
         */
        //INDArray postOutput = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFn, preOutput.dup()));
        INDArray postOutput = activationFn.getActivation(preOutput.dup(),true);

        INDArray yhatmag = postOutput.norm2(1);
        INDArray ymag = labels.norm2(1);

        INDArray scoreArr = postOutput.mul(labels);
        scoreArr.diviColumnVector(yhatmag);
        scoreArr.diviColumnVector(ymag);

        if (exampleWeights != null) {
            if (exampleWeights.length() != scoreArr.size(0)) {
                throw new IllegalStateException("Example Weights vector (length " + exampleWeights.length() + ") does not match scoreArr.size(0)=" + scoreArr.size(0));
            }
            scoreArr.muliColumnVector(exampleWeights);
        }

        if (mask != null) scoreArr.muliColumnVector(mask);
        return scoreArr.muli(-1);
    }

    @Override
    public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, INDArray exampleWeights, boolean average) {
        INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask, exampleWeights);

        double score = scoreArr.sumNumber().doubleValue();

        if(average) score /= scoreArr.size(0);

        return score;
    }

    @Override
    public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, INDArray exampleWeights) {
        INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask, exampleWeights);
        return scoreArr.sum(1);
    }

    @Override
    public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, INDArray exampleWeights) {
        INDArray yhat = activationFn.getActivation(preOutput.dup(),true);
        INDArray yL2norm = labels.norm2(1);

        INDArray yhatL2norm = yhat.norm2(1);
        INDArray yhatL2normSq = yhatL2norm.mul(yhatL2norm);

        //Note: This is not really the L1 norm since I am not taking abs values
        INDArray yhatDotyL1norm = labels.mul(yhat).sum(1);

        INDArray dLda = labels.mulColumnVector(yhatL2normSq);
        dLda.subi(yhat.mulColumnVector(yhatDotyL1norm));
        dLda.diviColumnVector(yL2norm);
        dLda.diviColumnVector(yhatL2norm.mul(yhatL2normSq));
        dLda.muli(-1);

        if (exampleWeights != null) {
            if (exampleWeights.length() != dLda.size(0)) {
                throw new IllegalStateException("Example Weights vector (length " + exampleWeights.length() + ") does not match dLda.size(0)=" + dLda.size(0));
            }
            dLda.muliColumnVector(exampleWeights);
        }

        //dL/dz
        INDArray gradients = activationFn.backprop(preOutput, dLda).getFirst();      //TODO loss functions with params

        if(mask != null){
            gradients.muliColumnVector(mask);
        }

        return gradients;
    }

    @Override
    public org.apache.commons.math3.util.Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, INDArray exampleWeights, boolean average) {
        //TODO: probably a more efficient way to do this...

        return new Pair<>(
                computeScore(labels, preOutput, activationFn, mask, exampleWeights, average),
                computeGradient(labels, preOutput, activationFn, mask, exampleWeights));
    }

    @Override
    public String toString(){
        return "LossCosineProximity()";
    }
}
