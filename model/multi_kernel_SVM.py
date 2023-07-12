# Multi kernel learning based on SVM

from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
import numpy as np


class multi_kernel_SVM (SVC):
    """
    Support Vector Classification with multiple kernels
    The implementation is based on sklearn's SVC

    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive. The penalty
        is a squared l2 penalty.

    kernel_type : {'linear', 'rbf'}
        default='linear'
        Specifies the kernel type to be used for each subparts.

    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.

        - if ``gamma='scale'`` (default) is passed then it uses
          1 / (n_features * X.var()) as value of gamma,
        - if 'auto', uses 1 / n_features
        - if float, must be non-negative.

    shrinking : bool, default=True
        Whether to use the shrinking heuristic.
        See the :ref:`User Guide <shrinking_svm>`.

    probability : bool, default=False
        Whether to enable probability estimates. This must be enabled prior
        to calling `fit`, will slow down that method as it internally uses
        5-fold cross-validation, and `predict_proba` may be inconsistent with
        `predict`. Read more in the :ref:`User Guide <scores_probabilities>`.

    tol : float, default=1e-3
        Tolerance for stopping criterion.

    cache_size : float, default=200
        Specify the size of the kernel cache (in MB).

    class_weight : dict or 'balanced', default=None
        Set the parameter C of class i to class_weight[i]*C for
        SVC. If not given, all classes are supposed to have
        weight one.
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

    verbose : bool, default=False
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in libsvm that, if enabled, may not work
        properly in a multithreaded context.

    max_iter : int, default=-1
        Hard limit on iterations within solver, or -1 for no limit.

    decision_function_shape : {'ovo', 'ovr'}, default='ovr'
        Whether to return a one-vs-rest ('ovr') decision function of shape
        (n_samples, n_classes) as all other classifiers, or the original
        one-vs-one ('ovo') decision function of libsvm which has shape
        (n_samples, n_classes * (n_classes - 1) / 2). However, note that
        internally, one-vs-one ('ovo') is always used as a multi-class strategy
        to train models; an ovr matrix is only constructed from the ovo matrix.
        The parameter is ignored for binary classification.

    break_ties : bool, default=False
        If true, ``decision_function_shape='ovr'``, and number of classes > 2,
        :term:`predict` will break ties according to the confidence values of
        :term:`decision_function`; otherwise the first class among the tied
        classes is returned. Please note that breaking ties comes at a
        relatively high computational cost compared to a simple predict.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo random number generation for shuffling the data for
        probability estimates. Ignored when `probability` is False.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    class_weight_ : ndarray of shape (n_classes,)
        Multipliers of parameter C for each class.
        Computed based on the ``class_weight`` parameter.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    coef_ : ndarray of shape (n_classes * (n_classes - 1) / 2, n_features)
        Weights assigned to the features (coefficients in the primal
        problem). This is only available in the case of a linear kernel.

        `coef_` is a readonly property derived from `dual_coef_` and
        `support_vectors_`.

    dual_coef_ : ndarray of shape (n_classes -1, n_SV)
        Dual coefficients of the support vector in the decision
        function (see :ref:`sgd_mathematical_formulation`), multiplied by
        their targets.
        For multiclass, coefficient for all 1-vs-1 classifiers.
        The layout of the coefficients in the multiclass case is somewhat
        non-trivial. See the :ref:`multi-class section of the User Guide
        <svm_multi_class>` for details.

    fit_status_ : int
        0 if correctly fitted, 1 otherwise (will raise warning)

    intercept_ : ndarray of shape (n_classes * (n_classes - 1) / 2,)
        Constants in decision function.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    n_iter_ : ndarray of shape (n_classes * (n_classes - 1) // 2,)
        Number of iterations run by the optimization routine to fit the model.
        The shape of this attribute depends on the number of models optimized
        which in turn depends on the number of classes.

    support_ : ndarray of shape (n_SV)
        Indices of support vectors.

    support_vectors_ : ndarray of shape (n_SV, n_features)
        Support vectors.

    n_support_ : ndarray of shape (n_classes,), dtype=int32
        Number of support vectors for each class.

    probA_ : ndarray of shape (n_classes * (n_classes - 1) / 2)
    probB_ : ndarray of shape (n_classes * (n_classes - 1) / 2)
        If `probability=True`, it corresponds to the parameters learned in
        Platt scaling to produce probability estimates from decision values.
        If `probability=False`, it's an empty array. Platt scaling uses the
        logistic function
        ``1 / (1 + exp(decision_value * probA_ + probB_))``
        where ``probA_`` and ``probB_`` are learned from the dataset [2]_. For
        more information on the multiclass case and training procedure see
        section 8 of [1]_.

    shape_fit_ : tuple of int of shape (n_dimensions_of_X,)
        Array dimensions of training vector ``X``.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from multi_kernel_SVM import multi_kernel_SVM
    >>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    >>> y = np.array([1, 1, 2, 2])
    >>> clf = make_pipeline(StandardScaler(), multi_kernel_SVM(w1=0.4, dims=[1,1]))
    >>> clf.fit(X, y)
    >>> print(clf.predict([[-0.8, -1]]))
    [1]
    """

    def __init__(self, *,
                 w1=0, w2=0, w3=0, w4=0, w5=0,
                 dims=None,
                 C=1.0,
                 kernel_type='linear',
                 gamma="scale",
                 shrinking=True,
                 probability=False,
                 tol=1e-3,
                 cache_size=200,
                 class_weight=None,
                 verbose=False,
                 max_iter=-1,
                 decision_function_shape="ovr",
                 break_ties=False,
                 random_state=None
                 ):

        # scikit-opt library does not accept 'Space' objects as hyperparam.
        # Only integer, real or categorical are allowed.
        # Till this is fixed, circumventing by passing each weight as a separate param rather than a vector/dict.
        # Only first len(dims)-many weights will be used, extra dimensions will be unused.
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.w5 = w5

        # dims contains the number of feature per each AE block.
        # convert to column number of their endpoints
        block_endpoints = [0, *np.cumsum(dims)]
        self.num_AE = len(block_endpoints) - 1
        self.dims = dims

        # The kernel function to be used for each SVM subunit of the model
        self.kernel_type = kernel_type
        if self.kernel_type == 'linear':
            def kernel_eval(a, b):
                return np.dot(a, b.T)
        elif self.kernel_type == 'rbf':
            def kernel_eval(a, b):
                if self.gamma == 'scale':
                    gamma = 1000
                else:
                    gamma = self.gamma
                return RBF(gamma)(a, b)
        else:
            raise Exception('Unimplemented kernel type %s' % self.kernel_type)

        def compute_kernel(x, y):
            # n-1 random points need to find n weights
            # And then you split the 0-1 interval into n segments
            angles = [self.w1, self.w2, self.w3, self.w4, self.w5][:self.num_AE - 1]

            # The successive spherical coordinates could have been evaluated by x_k = x_k-1 * tan() * cos()
            # But opting for analytical evaluation here for numerical stability
            # sin(a)*sin(b)*...*sin(y)*cos(z)
            spherical_unit_vec = []
            for space_dim in range(len(angles) + 1):
                term = 1
                for j in range(0, space_dim):
                    term *= np.sin(np.pi / 2 * angles[j])

                if space_dim < len(angles):
                    # All coordinates x1, x2, x3, ... x_(n-1) except x_n are:
                    # sin(a)*sin(b)*...*sin(y)*cos(z)
                    term *= np.cos(np.pi / 2 * angles[space_dim])

                spherical_unit_vec.append(term)

            # "weights" is a point on x+y+z+... = 1 plane
            # At the end, weights should sum up to 1.
            r = 1 / np.sum(spherical_unit_vec)
            weights = r * np.array(spherical_unit_vec)

            computed_kernel = np.zeros((x.shape[0], y.shape[0]))
            for i in range(self.num_AE):
                x_subset = x[:, block_endpoints[i]:block_endpoints[i + 1]]
                y_subset = y[:, block_endpoints[i]:block_endpoints[i + 1]]
                computed_kernel += weights[i] * kernel_eval(x_subset, y_subset)
                return computed_kernel

        super().__init__(
            kernel=compute_kernel,
            gamma=gamma,
            tol=tol,
            C=C,
            shrinking=shrinking,
            probability=probability,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state
        )
