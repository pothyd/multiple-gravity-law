from itertools import product
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm


class gravity_model(object):
    def __init__(
        self, data_dir: str, country: str, city: str, cost: str = "rij"
    ) -> None:
        """Gravity model

        Args:
            data_dir (str): data directory
            country (str): data country
            city (str): data city
            cost (str): cost variable. Defaults to "rij" (distance).
        """
        self.meta_data = {"country": country, "city": city}

        self.cost = cost

        self.load_data(data_dir)

    def load_data(self, data_dir: str) -> None:
        """Load dataset

        Args:
            data_dir (str): data directory
        """
        # load mobility data
        # distance as a cost function
        if self.cost == "rij":
            self.mob_data_all = pd.read_csv(
                "{}/{}/MOB_{}.csv".format(
                    data_dir, self.meta_data["country"], self.meta_data["city"]
                )
            )
        # simulated cost as a cost functio
        elif self.cost == "cij":
            self.mob_data_all = pd.read_csv(
                "{}/{}_simul/SIM_{}.csv".format(
                    data_dir, self.meta_data["country"], self.meta_data["city"]
                )
            )
            self.mob_data_all["cij"] = self.mob_data_all.cij / self.mob_data_all.rij
        
        # load population data
        self.pop_data_all = pd.read_csv(
            "{}/{}/POP_{}.csv".format(
                data_dir, self.meta_data["country"], self.meta_data["city"]
            )
        )
        self.mob_data_all = self.mob_data_all.drop(["ti", "tj"], axis=1)
        self.pop_data_all = self.pop_data_all.drop(["out_travel", "in_travel"], axis=1)

        # coordinate setting
        self.pop_data_all = self.pop_data_all.astype({"X": "int", "Y": "int"})
        min_coords = {"X": min(self.pop_data_all.X), "Y": min(self.pop_data_all.Y)}
        self.pop_data_all["X"] = self.pop_data_all.X - min_coords["X"]
        self.pop_data_all["Y"] = self.pop_data_all.Y - min_coords["Y"]

        # symmetrize US commuting dataset
        if self.meta_data["country"] == "US":
            self.symmetrize_data()

    def symmetrize_data(self) -> None:
        """Symmetrize data"""
        # Symmetrize mobility data
        self.mob_data_all = pd.concat(
            [
                self.mob_data_all,
                self.mob_data_all.rename(columns={"o": "d", "d": "o"}),
            ],
            sort=True,
        ).reset_index(drop=True)
        self.mob_data_all = self.mob_data_all.groupby(
            by=["o", "d"], as_index=False
        ).sum()

        # Symmetrize population data
        a = (
            self.mob_data_all[["o"]]
            .drop_duplicates(subset=["o"])
            .rename(columns={"o": "id"})
        )
        b = (
            self.mob_data_all[["d"]]
            .drop_duplicates(subset=["d"])
            .rename(columns={"d": "id"})
        )
        self.pop_data_all = self.pop_data_all[["id", "X", "Y"]].merge(a, on="id")
        self.pop_data_all = self.pop_data_all.merge(b, on="id")
        self.pop_data_all = self.pop_data_all.dropna()

    def exponent_matrix(
        self,
        rmin: float = 0,
        rmax: float = 1000.0,
        mbins: int = 10,
        rbins: int = 100,
        bin_mode: str = "equal_mass",
        binning: bool = True,
    ) -> None:
        """Generate exponent matrix

        Args:
            rmin (float, optional): minimum value of distance. Defaults to 0.
            rmax (float, optional): maximum value of distance. Defaults to 1000.0.
            mbins (int, optional): number of traffic bin. Defaults to 10.
            rbins (int, optional): number of distance bin. Defaults to 100.
            bin_mode (str, optional): binning method. Defaults to "equal_mass".
            binning (bool, optional): _description_. Defaults to True.
        """

        self.preprocess_data(rmin, rmax)
        self.divide_bins(mbins, rbins, bin_mode)
        self.calculate_matrix(mbins, rbins, binning)

    def preprocess_data(self, rmin: float, rmax: float):
        """Preprocess data
        
        Args:
            rmin (float, optional): minimum value of distance.
            rmax (float, optional): maximum value of distance.
        """
        # Distance limit: between RMIN and RMAX
        self.mob_data = self.mob_data_all[
            (self.mob_data_all[self.cost] > rmin)
            & (self.mob_data_all[self.cost] < rmax)
        ]

        # Recalculate out traffic, in traffic, mass
        self.pop_data = pd.merge(
            self.pop_data_all,
            self.mob_data.groupby("o")
            .sum()[["tij"]]
            .reset_index()
            .rename(columns={"o": "id", "tij": "out_travel"}),
            on="id",
            how="left",
        )
        self.pop_data = pd.merge(
            self.pop_data,
            self.mob_data.groupby("d")
            .sum()[["tij"]]
            .reset_index()
            .rename(columns={"d": "id", "tij": "in_travel"}),
            on="id",
            how="left",
        )
        self.pop_data = self.pop_data.fillna(0)
        self.pop_data["mass"] = self.pop_data.out_travel + self.pop_data.in_travel
        self.pop_data = self.pop_data[self.pop_data.mass != 0]

        self.mob_data = pd.merge(
            self.mob_data,
            self.pop_data[["id", "mass"]].rename(columns={"id": "o", "mass": "ti"}),
            on="o",
            how="left",
        )
        self.mob_data = pd.merge(
            self.mob_data,
            self.pop_data[["id", "mass"]].rename(columns={"id": "d", "mass": "tj"}),
            on="d",
            how="left",
        )

        # Rescaled Tij: Tij/(TiTj)
        self.mob_data["rescaled_tij"] = self.mob_data.tij / (
            self.mob_data.ti + self.mob_data.tj
        )

        self.calculate_metadata()

    def calculate_metadata(self) -> None:
        """Calculate metadata"""
        self.meta_data["total_mass"] = self.pop_data.mass.sum()
        self.meta_data["mean_distance"] = (
            np.sum(self.mob_data[self.cost] * self.mob_data.tij)
            / self.mob_data.tij.sum()
        )

    def divide_bins(self, mbins: int, rbins: int, bin_mode="equal_mass"):
        """Divide population bins and distance bins
        
        Args:
            mbins (int, optional): number of traffic bin.
            rbins (int, optional): number of distance bin.
            bin_mode (str, optional): binning method. Defaults to "equal_mass".
        """
        # Population binning
        self.pop_data = self.pop_data.sort_values(by=["mass", "id"]).reset_index(
            drop=True
        )

        if bin_mode == "equal_len":
            bin_space = np.linspace(
                min(self.pop_data.mass), max(self.pop_data.mass) + 1, mbins + 1
            )
            self.pop_data["bin"] = [
                np.mean(
                    [
                        i
                        for i in range(mbins)
                        if m >= bin_space[i] and m < bin_space[i + 1]
                    ]
                )
                for m in self.pop_data.mass
            ]

        elif bin_mode == "equal_num":
            self.pop_data["bin"] = [
                int(mbins * i / len(self.pop_data)) for i in range(len(self.pop_data))
            ]

        elif bin_mode == "equal_mass":
            cur_bin = 0
            bin_set = []
            cum_mass = 0
            tot_mass = sum(self.pop_data.mass)
            for r in self.pop_data.itertuples():
                bin_set.append(cur_bin)
                if cum_mass > (cur_bin + 1) * tot_mass / mbins:
                    cur_bin += 1
                cum_mass += r.mass

            self.pop_data["bin"] = bin_set

        # Distance binning
        self.bin_distance = np.logspace(
            np.log10(min(self.mob_data[self.cost])),
            np.log10(max(self.mob_data[self.cost])),
            num=rbins + 1,
        )

    def calculate_matrix(self, mbins: int, rbins: int, binning: bool = True):
        """Calculate matrix
        
        Args:
            mbins (int, optional): number of traffic bin.
            rbins (int, optional): number of distance bin.
            binning (bool, optional): _description_. Defaults to True.
        """
        self.matrix = {col: np.empty((mbins, mbins)) for col in ["gamma", "rsq", "num", "dist_dis", "dist_avg", "dist_var", "dist_std", "dist_fra"]}
#         self.matrix["gamma"] = np.empty((mbins, mbins))
#         self.matrix["rsq"] = np.empty((mbins, mbins))
#         self.matrix["num"] = np.zeros((mbins, mbins))
#         self.matrix["dist_dis"] = np.zeros((mbins, mbins))
#         self.matrix["dist_avg"] = np.zeros((mbins, mbins))
#         self.matrix["dist_var"] = np.zeros((mbins, mbins))
#         self.matrix["dist_std"] = np.zeros((mbins, mbins))
#         self.matrix["dist_fra"] = np.zeros((mbins, mbins))

        for p in tqdm(product(range(mbins), repeat=2), desc="Calculating exponent"):
            origin = self.pop_data[self.pop_data.bin == p[0]].id.tolist()
            destin = self.pop_data[self.pop_data.bin == p[1]].id.tolist()
            df = self.mob_data[
                (self.mob_data.o.isin(origin)) & (self.mob_data.d.isin(destin))
            ]
            if sum(df.tij) != 0:
                dist_dis = np.mean(df.rij)
                dist_avg = sum(df.rij * df.tij) / sum(df.tij)
                dist_var = sum((df.rij - dist_avg) ** 2 * df.tij) / sum(df.tij)
                dist_std = np.sqrt(dist_var)

                self.matrix["dist_dis"][p[0], p[1]] = dist_dis
                self.matrix["dist_avg"][p[0], p[1]] = dist_avg
                self.matrix["dist_var"][p[0], p[1]] = dist_var
                self.matrix["dist_std"][p[0], p[1]] = dist_std
                self.matrix["dist_fra"][p[0], p[1]] = dist_avg / dist_dis
            else:
                self.matrix["dist_dis"][p[0], p[1]] = 0
                self.matrix["dist_avg"][p[0], p[1]] = 0
                self.matrix["dist_var"][p[0], p[1]] = 0
                self.matrix["dist_std"][p[0], p[1]] = 0
                self.matrix["dist_fra"][p[0], p[1]] = 0

            mobility_pair = self.mob_data[
                (self.mob_data.o.isin(origin)) & (self.mob_data.d.isin(destin))
            ][[self.cost, "rescaled_tij"]]

            if len(mobility_pair) > 0:
                exponent, rsq = self.calculate_exponent(mobility_pair, rbins, binning)
                if True:  # exponent < 1 and exponent > -10:
                    self.matrix["gamma"][p[0], p[1]] = exponent
                    self.matrix["rsq"][p[0], p[1]] = rsq
                else:
                    self.matrix["gamma"][p[0], p[1]] = np.nan
                    self.matrix["rsq"][p[0], p[1]] = 0
                self.matrix["num"][p[0], p[1]] = np.log10(len(mobility_pair))

    def calculate_exponent(self, mobility_pair: pd.DataFrame, rbins: int = 100, binning: bool = True):
        """Calculate distance exponent value
        
        Args:
            mobility_pair (pd.DataFrame): mobility pairs.
            rbins (int, optional): number of distance bin. Defaults to 100.
            binning (bool, optional): _description_. Defaults to True.
        """
        
        df = mobility_pair.sort_values(by=self.cost)

        if binning:
            df["bin"] = pd.cut(
                df[self.cost], bins=self.bin_distance, labels=range(rbins)
            )
            df = df.groupby("bin").mean().dropna()
            df = pd.merge(
                df,
                pd.DataFrame(
                    {
                        "bin": range(rbins),
                        "rij_bin": [
                            (self.bin_distance[i] + self.bin_distance[i + 1]) / 2
                            for i in range(rbins)
                        ],
                    }
                ),
                on="bin",
                how="left",
            )
            df = df[(df.rij_bin != 0) & (df.rescaled_tij != 0)]
            X = np.log10(df.rij_bin)
            Y = np.log10(df.rescaled_tij)
        else:
            df = df[(df.rij != 0) & (df.rescaled_tij != 0)]
            X = np.log10(df.rij)
            Y = np.log10(df.rescaled_tij)

        X = sm.add_constant(X)

        result = sm.OLS(Y, X).fit()

        if len(result.params) > 1:
            exponent = -result.params[1]
            rsq = result.rsquared
            conf_int = result.conf_int(0.01)[1]
        else:
            exponent = np.nan
            rsq = np.nan

        if rsq > 1:

            plt.scatter(
                mobility_pair.rij,
                mobility_pair.rescaled_tij,
                facecolor="none",
                edgecolor="k",
                alpha=0.3,
            )
            plt.scatter(X, Y, facecolor="none", edgecolor="r")
            plt.xscale("log")
            plt.yscale("log")
            plt.show()

        return exponent, rsq

    def calculate_average_matrix(self, col, mbins):
        self.matrix[col] = np.zeros((mbins, mbins))

        for p in tqdm(product(range(mbins), repeat=2)):
            origin = self.pop_data[self.pop_data.bin == p[0]].id.tolist()
            destin = self.pop_data[self.pop_data.bin == p[1]].id.tolist()
            mobility_pair = self.mob_data[
                (self.mob_data.o.isin(origin)) & (self.mob_data.d.isin(destin))
            ][[self.cost, "rescaled_tij"]]

            if len(mobility_pair) > 0:
                self.matrix[col][p[0], p[1]] = np.median(mobility_pair[col])
            else:
                self.matrix[col][p[0], p[1]] = 0

    def plot_maps(
        self,
        col: str,
        scale_bar_x: int = 45,
        scale_bar_y: int = 25,
        scale_bar_l=30,
        x_size: int = 7,
        path: str = None,
        cmap="inferno",
    ):
        x_max = max(self.pop_data.X) - min(self.pop_data.X)
        y_max = max(self.pop_data.Y) - min(self.pop_data.Y)

        map = pd.DataFrame(
            index=range(0, y_max), columns=range(0, x_max), dtype="float64"
        )

        for i in range(max(self.pop_data.bin) + 1):
            for r in self.pop_data[self.pop_data[col] == i].itertuples():
                map.at[r.Y, r.X] = i

        plt.rcParams["figure.figsize"] = (x_size, x_size * y_max / x_max)

        # Plot map
        plt.pcolor(map, cmap=cmap, edgecolors="none")

        # Plot scale bar
        plt.plot(
            [scale_bar_x, scale_bar_x + scale_bar_l],
            [scale_bar_y, scale_bar_y],
            "k",
        )
        # plt.annotate(
        #     "{} km".format(scale_bar_l),
        #     xy=(scale_bar_x + scale_bar_l / 2, scale_bar_y - 10),
        #     fontsize=15,
        #     ha="center",
        # )

        plt.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
        plt.tick_params(
            axis="y", which="both", left=False, right=False, labelleft=False
        )
        plt.axis("off")
        if path != None:
            plt.savefig(
                "{}/landscape_{}.pdf".format(path, self.meta_data["city"]),
                bbox_inches="tight",
            )
        plt.show()

    def plot_matrix(self, fig, ax, param="gamma", path=None, ext_val=None, ref=None):

        if ext_val == None:
            ext_val = [self.matrix[param].min(), self.matrix[param].max()]

        mbins = max(self.pop_data.bin) + 1
        im = ax.imshow(
            self.matrix[param],
            origin="lower",
            cmap="viridis",
            vmin=ext_val[0],
            vmax=ext_val[1],
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=24)
        cbar.ax.set_yticklabels(["{:.1f}".format(i) for i in cbar.get_ticks()])
        if ref != None:
            cbar.ax.axhline(y=ref, c="w", linewidth=2)
            # cbar.set_ticks(cbar.get_ticks() + [ref])
            # cbar.ax.set_yticklabels(cbar.ax.get_yticklabels() + ['Ref'])

        ticks = np.linspace(0, mbins - 1, mbins)
        ticklabels_i = ["{}".format(i + 1) for i in range(mbins)]
        ticklabels_j = ["{}".format(i + 1) for i in range(mbins)]
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels_i, rotation=0, fontsize=28, family="Arial")
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticklabels_j, fontsize=28, family="Arial")
        ax.set_xlabel("$k$", fontsize=34)
        ax.set_ylabel("$k'$", fontsize=34)
        if path != None:
            plt.savefig(
                "{}/{}_{}.pdf".format(path, param, self.meta_data["city"]),
                bbox_inches="tight",
            )
