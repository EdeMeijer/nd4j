package org.nd4j.linalg.lossfunctions.impl;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Sign;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.serde.RowVectorDeserializer;
import org.nd4j.linalg.lossfunctions.serde.RowVectorSerializer;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;
import org.nd4j.shade.jackson.databind.annotation.JsonSerialize;

/**
 * L1 loss function: i.e., sum of absolute errors, L = sum_i abs(predicted_i - actual_i)
 * See also {@link LossMAE} for a mathematically similar loss function (MAE has division by N, where N is output size)
 *
 * @author Susan Eraly
 */
@EqualsAndHashCode
@JsonInclude(JsonInclude.Include.NON_NULL)
@Getter
public class LossL1 implements ILossFunction {

    @JsonSerialize(using = RowVectorSerializer.class)
    @JsonDeserialize(using = RowVectorDeserializer.class)
    protected final INDArray weights;

    public LossL1() {
        this(null);
    }

    /**
     * L1 loss function where each the output is (optionally) weighted/scaled by a fixed scalar value.
     * Note that the weights array must be a row vector, of length equal to the labels/output dimension 1 size.
     * A weight vector of 1s should give identical results to no weight vector.
     *
     * @param weights Weights array (row vector). May be null.
     */
    public LossL1(INDArray weights) {
        if (weights != null && !weights.isRowVector()) {
            throw new IllegalArgumentException("Weights array must be a row vector");
        }
        this.weights = weights;
    }

    public INDArray scoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, INDArray exampleWeights) {
        INDArray scoreArr;
        //INDArray output = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFn, preOutput.dup()));
        INDArray output = activationFn.getActivation(preOutput.dup(),true);
        scoreArr = output.subi(labels);
        Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("abs", scoreArr));

        //Weighted loss function
        if (weights != null) {
            if (weights.length() != scoreArr.size(1)) {
                throw new IllegalStateException("Weights vector (length " + weights.length() + ") does not match scoreArr.size(1)=" + scoreArr.size(1));
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
        INDArray output = activationFn.getActivation(preOutput.dup(),true);

        INDArray outSubLabels = output.sub(labels);
        INDArray dLda = Nd4j.getExecutioner().execAndReturn(new Sign(outSubLabels));

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
        INDArray gradients = activationFn.backprop(preOutput, dLda).getFirst();  //TODO activation function param gradients

        if (mask != null) {
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
    public String toString() {
        if (weights == null) return "LossL1()";
        return "LossL1(weights=" + weights + ")";
    }
}
