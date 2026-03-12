import { Mat3 } from '../types/index.js';

interface EigenResult {
    eigenvalues: [number, number, number];
    V: Float64Array;
}
interface SVDResult {
    U: Float64Array;
    S: [number, number, number];
    V: Float64Array;
}
/** Jacobi eigendecomposition for 3x3 symmetric matrix */
declare function symmetricEigen3x3(S: Mat3): EigenResult;
/** SVD for 3x3: H = U * diag(S) * V^T */
declare function svd3x3(H: Mat3): SVDResult;

export { type EigenResult, type SVDResult, svd3x3, symmetricEigen3x3 };
