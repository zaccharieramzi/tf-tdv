import tensorflow as tf
from tensorflow.keras.models import Model


class UnrolledFB(Model):
    """The unrolled Forward-Backward model with any kind of prox.
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
            self.original_prox = model_class(**model_kwargs)
        self.proxs = [
            self.original_prox if self.weight_sharing else model_class(**model_kwargs)
            for _ in range(self.n_iter)
        ]
        self.alphas = [  # equivalent of T/S
            self.add_weight(
                shape=(1,),
                initializer=tf.keras.initializers.constant(self.init_step_size),
                name=f'alpha_{i}',
            )
            for i in range(self.n_iter)
        ]
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
        for alpha, prox in zip(self.alphas, self.proxs):
            new_image = current_image - alpha * self.grad(current_image, inputs)
            new_image = prox(new_image)
            # NOTE: we use this decorrelation of var names to allow for
            # Nesterov implementation
            current_image = new_image
        return current_image
