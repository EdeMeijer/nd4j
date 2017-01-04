package org.nd4j.linalg.lossfunctions.impl;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.serde.RowVectorDeserializer;
import org.nd4j.linalg.lossfunctions.serde.RowVectorSerializer;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;
import org.nd4j.shade.jackson.databind.annotation.JsonSerialize;

/**
 * Mean Squared Logarithmic Error loss function: L = 1/N sum_i (log(1+predicted_i) - log(1+actual_i))^2
 *
 * @author Susan Eraly
 */
@EqualsAndHashCode
@JsonInclude(JsonInclude.Include.NON_NULL)
@Getter
public class LossMSLE implements ILossFunction {

    @JsonSerialize(using = RowVectorSerializer.class)
    @JsonDeserialize(using = RowVectorDeserializer.class)
    private final INDArray weights;

    public LossMSLE() {
        this(null);
    }

    /**
     * Mean Squared Logarithmic Error loss function where each the output is (optionally) weighted/scaled by a fixed scalar value.
     * Note that the weights array must be a row vector, of length equal to the labels/output dimension 1 size.
     * A weight vector of 1s should give identical results to no weight vector.
     *
     * @param weights Weights array (row vector). May be null.
     */
    public LossMSLE(INDArray weights) {
        if (weights != null && !weights.isRowVector()) {
            throw new IllegalArgumentException("Weights array must be a row vector");
        }
        this.weights = weights;
    }

    public INDArray scoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, INDArray exampleWeights) {
        INDArray scoreArr;
        //INDArray output = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFn, preOutput.dup()));
        INDArray output = activationFn.getActivation(preOutput.dup(),true);
        scoreArr = Transforms.log(output.addi(1.0).divi(labels.add(1.0)), false);
        scoreArr = scoreArr.muli(scoreArr).divi(labels.size(1));

        //Weighted loss function
        if (weights != null) {
            if (weights.length() != output.size(1)) {
                throw new IllegalStateException("Weights vector (length " + weights.length() + ") does not match output.size(1)=" + output.size(1));
            }
            scoreArr.muliRowVector(weights);
        }
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
        //INDArray output = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFn, preOutput.dup()));
        INDArray output = activationFn.getActivation(preOutput.dup(),true);

        INDArray p1 = output.add(1.0);
        INDArray dLda = p1.rdiv(2.0 / labels.size(1));
        INDArray logRatio = Transforms.log(p1.divi(labels.add(1.0)), false);
        dLda.muli(logRatio);

        if (weights != null) {
            if (weights.length() != dLda.size(1)) {
                throw new IllegalStateException("Weights vector (length " + weights.length() + ") does not match dLda.size(1)=" + dLda.size(1));
            }
            dLda.muliRowVector(weights);
        }
        if (exampleWeights != null) {
            if (exampleWeights.length() != dLda.size(0)) {
                throw new IllegalStateException("Example Weights vector (length " + exampleWeights.length() + ") does not match dLda.size(0)=" + dLda.size(0));
            }
            dLda.muliColumnVector(exampleWeights);
        }

        //dL/dz
        INDArray gradients = activationFn.backprop(preOutput, dLda).getFirst(); //TODO activation functions with weights

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
        if (weights == null) return "LossMSLE()";
        return "LossMSLE(weights=" + weights + ")";
    }

}
