import tensorflow as tf
from tensorflow.keras.models import Model


class UnrolledFB(Model):
    """The unrolled Forward-Backward model with any kind of regularizer gradient.
    """
    def __init__(
            self,
            model_class,
            model_kwargs,
            inverse_problem='denoising',
            weight_sharing=True,
            n_iter=10,
            init_step_size=0.5,
            **kwargs,
        ):
        super().__init__(**kwargs)
        self.inverse_problem = inverse_problem
        self.weight_sharing = weight_sharing
        self.n_iter = n_iter
        self.init_step_size = init_step_size
        if self.weight_sharing:
            self.original_reg_grad = model_class(**model_kwargs)
        self.reg_grads = [
            self.original_reg_grad if self.weight_sharing else model_class(**model_kwargs)
            for _ in range(self.n_iter)
        ]
        # I thought that the 2 step sizes were common
        # but in the code it appears that they are not:
        # https://github.com/VLOGroup/tdv/blob/master/model.py#L111
        self.alpha = self.add_weight(
            shape=(1,),
            initializer=tf.keras.initializers.constant(self.init_step_size),
            constraint=tf.keras.constraints.NonNeg(),
            name='alpha',
        )
        self.lamda = self.add_weight(
            shape=(1,),
            initializer=tf.keras.initializers.constant(self.init_step_size),
            constraint=tf.keras.constraints.NonNeg(),
            name='lambda',
        )
        if self.inverse_problem == 'denoising':
            self.measurements_operator = lambda x: x
            self.measurements_operator_adjoint = lambda x: x
        else:
            raise NotImplementedError(f'{self.inverse_problem} is not implemented yet')


    def grad(self, image, inputs):
        gr = self.measurements_operator_adjoint(self.measurements_operator(image) - inputs)
        return gr


    def call(self, inputs):
        current_image = inputs
        for reg_grad in self.reg_grads:
            reg_grad_eval = self.alpha * reg_grad(current_image)
            grad_eval = self.lamda * self.grad(current_image, inputs)
            new_image = current_image - grad_eval - reg_grad_eval
            # NOTE: we use this decorrelation of var names to allow for
            # Nesterov implementation
            current_image = new_image
        return current_image
