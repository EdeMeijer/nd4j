package org.nd4j.linalg.lossfunctions.impl;


import lombok.EqualsAndHashCode;
import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Kullback Leibler Divergence loss function
 *
 * @author Susan Eraly
 */
@EqualsAndHashCode
public class LossKLD implements ILossFunction {

    private INDArray scoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, INDArray exampleWeights) {
        //INDArray output = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFn, preOutput.dup()));
        INDArray output = activationFn.getActivation(preOutput.dup(),true);

        // Clip output and labels to be between Nd4j.EPS_THREsHOLD and 1, i.e. a valid non-zero probability
        output = Transforms.min(Transforms.max(output, Nd4j.EPS_THRESHOLD, false), 1, false);
        labels = Transforms.min(Transforms.max(labels, Nd4j.EPS_THRESHOLD, true), 1, false);

        INDArray logRatio = Transforms.log(output.rdivi(labels), false);

        INDArray scoreArr = logRatio.muli(labels);
        if (exampleWeights != null) {
            if (exampleWeights.length() != scoreArr.size(0)) {
                throw new IllegalStateException("Example Weights vector (length " + exampleWeights.length() + ") does not match scoreArr.size(0)=" + scoreArr.size(0));
            }
            scoreArr.muliColumnVector(exampleWeights);
        }
        if (mask != null) {
            scoreArr.muliColumnVector(mask);
        }
        return scoreArr;
    }

    @Override
    public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, INDArray exampleWeights, boolean average) {
        INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask, exampleWeights);

        double score = scoreArr.sumNumber().doubleValue();

        if (average) {
            score /= scoreArr.size(0);
        }

        return score;
    }

    @Override
    public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, INDArray exampleWeights) {
        INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask, exampleWeights);
        return scoreArr.sum(1);
    }

    @Override
    public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, INDArray exampleWeights) {
        //INDArray output = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFn, preOutput.dup()));
        INDArray output = activationFn.getActivation(preOutput.dup(),true);

        INDArray dLda = labels.div(output).negi();
        if (exampleWeights != null) {
            if (exampleWeights.length() != dLda.size(0)) {
                throw new IllegalStateException("Example Weights vector (length " + exampleWeights.length() + ") does not match dLda.size(0)=" + dLda.size(0));
            }
            dLda.muliColumnVector(exampleWeights);
        }
        INDArray grad = activationFn.backprop(preOutput, dLda).getFirst();      //TODO activation functions with params

        if (mask != null) {
            grad.muliColumnVector(mask);
        }

        return grad;
    }

    @Override
    public Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, INDArray exampleWeights, boolean average) {
        //TODO: probably a more efficient way to do this...

        return new Pair<>(
                computeScore(labels, preOutput, activationFn, mask, exampleWeights, average),
                computeGradient(labels, preOutput, activationFn, mask, exampleWeights));
    }


    @Override
    public String toString() {
        return "LossKLD()";
    }
}
