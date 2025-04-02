import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    def __init__(self, iterations_range, step, repetitions, max_it =500):
        self.max_it = max_it
        self.iterations_range = iterations_range
        self.step = step
        self.repetitions = repetitions

    def plot_time_precision(self, algorithms, data_generator, test_function, save_path=None, legends=None, log_x_time=False, log_y_time=False, log_y_precision=False, log_x_precision=False,
                           x_label='Taille des données d\'entrée', y_label_time='Temps de calcul (s)', y_label_precision='Précision',
                           title_time='Temps de calcul vs Taille des données d\'entrée', title_precision='Précision vs Taille des données d\'entrée'):
        a, b = self.iterations_range
        iterations = np.arange(a, b + 1, self.step)

        all_times = {algo_name: [] for algo_name in algorithms}
        all_precisions = {algo_name: [] for algo_name in algorithms}

        for it in iterations:
            for algo_name, algo_func in algorithms.items():
                total_time = 0
                total_precision = 0

                for _ in range(self.repetitions):
                    ok = False
                    no_data = 0
                    it = self.max_it
                    while (not ok) and (it!=0):
                        it-=1
                        data = data_generator(it)
                        try :
                            precision, elapsed_time = test_function(algo_func, data)
                            ok = True
                        except :
                            precision, elapsed_time = 0
                            ok = False

                    total_time += elapsed_time
                    total_precision += precision
                if (self.repetitions - it) != 0 :
                    avg_time = total_time / (self.repetitions - it)
                    avg_precision = total_precision / (self.repetitions - it)
                else :
                    avg_time = None
                    avg_precision = None

                all_times[algo_name].append(avg_time)
                all_precisions[algo_name].append(avg_precision)

            # Sauvegarder l'évolution des courbes à chaque itération
            if save_path:
                self.__save_plot(iterations[:len(all_times[algo_name])], all_times, all_precisions, legends, save_path, it, x_label, y_label_time, y_label_precision, title_time, title_precision, log_x_time, log_y_time, log_y_precision, log_x_precision)

            # Nettoyer les ressources après chaque itération
            self.__clean_up()

        if not save_path:
            self.__construct_plot(iterations, all_times, all_precisions, legends, x_label, y_label_time, y_label_precision, title_time, title_precision, log_x_time, log_y_time, log_y_precision, log_x_precision)
            plt.show()

    def __save_plot(self, iterations, all_times, all_precisions, legends, save_path, iteration, x_label, y_label_time, y_label_precision, title_time, title_precision, log_x_time, log_y_time, log_y_precision, log_x_precision):
        fig, _ = self.__construct_plot(iterations, all_times, all_precisions, legends, x_label, y_label_time, y_label_precision, title_time, title_precision, log_x_time, log_y_time, log_y_precision, log_x_precision)

        # Sauvegarder le graphique à chaque itération
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Graphique sauvegardé à {save_path}")

    def __construct_plot(self, iterations, all_times, all_precisions, legends, x_label, y_label_time, y_label_precision, title_time, title_precision, log_x_time, log_y_time, log_y_precision, log_x_precision):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

        for i, (algo_name, _) in enumerate(all_times.items()):
            label = legends[i] if legends and i < len(legends) else algo_name
            ax1.plot(iterations[:len(all_times[algo_name])], all_times[algo_name], label=label)
            ax2.plot(iterations[:len(all_precisions[algo_name])], all_precisions[algo_name], label=label)

        ax1.set_xlabel(x_label)
        ax1.set_ylabel(y_label_time)
        ax1.set_title(title_time)
        if log_x_time:
            ax1.set_xscale('log')
        if log_y_time:
            ax1.set_yscale('log')
        ax1.legend()

        ax2.set_xlabel(x_label)
        ax2.set_ylabel(y_label_precision)
        ax2.set_title(title_precision)
        if log_x_precision:
            ax2.set_xscale('log')
        if log_y_precision:
            ax2.set_yscale('log')
        ax2.legend()

        plt.tight_layout()
        return fig, (ax1, ax2)

    def __clean_up(self):
        # plt.clf()
        # plt.close('all')
        # import gc
        # gc.collect()
        pass
