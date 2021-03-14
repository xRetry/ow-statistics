import time
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DataFrame:
    _zone_id: str or None
    _data: pd.DataFrame or None
    _outfits: Dict[str, None or str]
    _loadouts: pd.DataFrame or None
    _weapons: pd.DataFrame or None
    _vehicles: pd.DataFrame or None
    _exp_ids: Dict[str, pd.DataFrame]

    def __init__(self, files):
        self._census_url = 'http://census.daybreakgames.com/get/ps2:v2/{}/?{}c:limit={}'
        self._loadouts = None
        self._weapons = None
        self._vehicles = None
        self._data = None
        self._zone_id = None
        self._theme = 'dark_background'
        self._themes = ['classic', 'dark_background', 'ggplot', 'seaborn']
        self._factions = {
            '1': 'VS',
            '2': 'NC',
            '3': 'TR',
            '0': 'NS'
        }
        self._outfits = {
            'VS': None,
            'NC': None,
            'TR': None
        }
        self._exp_ids = {}
        self._outfits_loaded = pd.DataFrame(columns=['outfit_tag', 'outfit_id', 'faction', 'players']).set_index('outfit_tag')
        self._colors = {
            'VS': 'purple',
            'NC': 'blue',
            'TR': 'red',
            'NS': 'grey'
        }

        self.set_theme(self._theme)
        self._from_json(files)

    def set_theme(self, theme_name):
        if theme_name in self._themes:
            plt.style.use(theme_name)
        else:
            raise AttributeError('Invalid Theme Name! ({})\nAvailable Themes: {}'.format(theme_name, self.available_themes))

    def set_match(self, zone_id: str = None, outfit_tag: str = None):
        # no input
        if zone_id is None and outfit_tag is None:
            raise AttributeError('Either zone_id or outfit_tag has to be provided!')
        # outfit tag as input -> get zone id from tag
        elif outfit_tag is not None:
            self._load_outfit(alias=outfit_tag)
            zone_ids_id = self._data[self._data.outfit_id == self._outfits_loaded.loc[outfit_tag].outfit_id].zone_id.unique()
            zone_ids_tag = self._data[self._data.outfit_id == outfit_tag].zone_id.unique()
            zone_ids = np.concatenate([zone_ids_id, zone_ids_tag])
            if len(zone_ids) == 0:
                raise KeyError('No matches from outfit \'{}\' found in dataset.'.format(outfit_tag))
            if len(zone_ids) > 1:
                raise LookupError('Multiple Zone IDs found for given outfit: {}\nUse zone id to find match instead.'.format(zone_ids))
            else:
                self.set_match(zone_id=zone_ids[0])
        # zone id as input
        elif zone_id in self.zone_ids:
            self.reset_match()
            self._zone_id = zone_id
            keys = self._outfits.keys()
            for i, k in enumerate(keys):
                if self._outfits[k] is None:
                    df = self._filter(new_faction_id=i+1)
                    outfit_id = df[(df.outfit_id != '0')].outfit_id.iloc[0]
                    outfit_tag = self._load_outfit(outfit_id=outfit_id)
                    self._outfits[k] = outfit_tag
            t_open, t_start, t_end = self._get_match_time()
            print('Times of loaded match:\nOpen: {}\nStart: {}\nEnd: {}'.format(t_open, t_start, t_end))
        else:
            raise AttributeError('Zone ID not found!\nAvailable Zone IDs: {}'.format(self.zone_ids))

    def reset_match(self):
        self._zone_id = None
        self._outfits = {
            'VS': None,
            'NC': None,
            'TR': None
        }

    def save_data(self, path='ow_data.json', selected_match=True):
        df = self._data
        if selected_match:
            df = self._filter()
        df.to_json(path)

    ''' Public Plotting Methods '''

    def plot_timeline_facility(self, figsize=(14, 5)):
        self._data_check()
        data_captures = self._filter(event_name='FacilityControl')
        t_open, t_start, t_end = self._get_match_time()
        n_bases = [[[0, 0]], [[0, 0]], [[0, 0]]]
        for i, row in data_captures[
            (data_captures.timestamp > t_open) &
            (data_captures.timestamp < t_end-pd.Timedelta('00:00:05'))
        ].iterrows():
            if int(row.new_faction_id) == 4: break
            if row.old_faction_id == row.new_faction_id: continue
            if int(row.old_faction_id) < 4:
                n_old_old = n_bases[int(row.old_faction_id) - 1][-1][0]
                n_old_new = n_bases[int(row.old_faction_id) - 1][-1][0] - 1
                n_bases[int(row.old_faction_id) - 1].append([
                    n_old_old,
                    (row.timestamp - t_start).total_seconds() / 60])
                n_bases[int(row.old_faction_id) - 1].append([
                    n_old_new,
                    (row.timestamp - t_start).total_seconds() / 60])

            n_new_old = n_bases[int(row.new_faction_id) - 1][-1][0]
            n_new_new = n_bases[int(row.new_faction_id) - 1][-1][0] + 1
            n_bases[int(row.new_faction_id) - 1].append([
                n_new_old,
                (row.timestamp - t_start).total_seconds() / 60])
            n_bases[int(row.new_faction_id) - 1].append([
                n_new_new,
                (row.timestamp - t_start).total_seconds() / 60])
        n_bases = [np.array(n_bases[0]), np.array(n_bases[1]), np.array(n_bases[2])]

        plt.figure(figsize=figsize)
        plt.plot([v[1] for v in n_bases[0]], [v[0] for v in n_bases[0]], c='purple')
        plt.plot([v[1] for v in n_bases[1]], [v[0] for v in n_bases[1]], c='blue')
        plt.plot([v[1] for v in n_bases[2]], [v[0] for v in n_bases[2]], c='red')

        plt.grid(True, alpha=0.2)
        plt.title('Territory Timeline')
        plt.xlabel('Elapsed Time [min]')
        plt.ylabel('Amount of Facilities')
        plt.show()

    def plot_timeline_kills(self, figsize=(14, 5)):
        self._data_check()
        self._plot_timeline(
            'Death',
            'Timeline of Kills',
            'Amount of Kills',
            column='attacker_character_id',
            figsize=figsize
        )

    def plot_timeline_deaths(self, with_revives=True, figsize=(14, 5)):
        self._data_check()
        event_name = 'Death'
        column = 'character_id'
        title = 'Timeline of Deaths'
        sub_idx = None
        if with_revives:
            event_name = [event_name, '*Revive']
            column = [column, column]
            title = title + ' (minus Revives)'
            sub_idx = [0, 1]
        self._plot_timeline(
            event_name,
            title,
            'Amount of Deaths',
            column=column,
            sub_idx=sub_idx,
            figsize=figsize
        )

    def plot_timeline_revives(self, figsize=(14, 5)):
        self._data_check()
        self._plot_timeline(
            '*Revive',
            'Timeline of Revives',
            'Amount of Revives',
            figsize=figsize
        )

    def plot_timeline_repair(self, figsize=(14, 5)):
        self._data_check()
        self._plot_timeline(
            '*Repair',
            'Timeline of Repair XP earned',
            'Amount of Repair XP',
            agg_fun='sum',
            figsize=figsize
        )

    def plot_timeline_vdestruction(self, figsize=(14, 5)):
        self._data_check()
        self._plot_timeline(
            '*Vehicle%20Destruction',
            'Timeline of Vehicle Kills',
            'Amount of Vehicle Kills',
            agg_fun='count',
            figsize=figsize
        )

    def plot_timeline_cortium(self, figsize=(14, 5)):
        self._data_check()
        self._plot_timeline(
            '*Cortium%20Deposit',
            'Timeline of Cortium Deposits',
            'Amount of Cortium Deposit XP',
            agg_fun='sum',
            figsize=figsize
        )

    def plot_timeline_kdr(self, with_revives=True, figsize=(14, 5)):
        self._data_check()
        event_name = ['Death', 'Death']
        column = ['attacker_character_id', 'character_id']
        title = 'Timeline of KDR'
        sub_idx = None
        div_idx = [0, 1]
        if with_revives:
            event_name = event_name + ['*Revive']
            column = column + ['character_id']
            title = title + ' (minus Revives)'
            sub_idx = [1, 2]
        self._plot_timeline(
            event_name,
            title,
            'Kills per Death',
            column=column,
            sub_idx=sub_idx,
            div_idx=div_idx,
            figsize=figsize)

    def plot_timeline_dpr(self, figsize=(14, 5)):
        self._data_check()
        self._plot_timeline(
            ['*Revive', 'Death'],
            'Proportion of deaths that got revived',
            'Revives per Deaths',
            column=['character_id', 'character_id'],
            div_idx=[0, 1],
            climit=[1000, 2],
            figsize=figsize
        )

    def plot_histogram_kills(self, bins=(70, 70, 70), figsize=(14, 5)):
        self._plot_hist('Kills', bins=bins, figsize=figsize)

    def plot_histogram_deaths(self, bins=(90, 90, 90), figsize=(14, 5)):
        self._plot_hist('Deaths', bins=bins, figsize=figsize)

    def plot_histogram_kdr(self, bins=(90, 90, 90), figsize=(14, 5)):
        self._plot_hist('KDR', bins=bins, figsize=figsize)

    def plot_weapon_kills(self, *factions):
        factions = self._verify_factions(factions)
        df = self._calc_weapon_stats(
            'attacker_weapon_id',
            'attacker_character_id',
            faction=factions,
            ids=self._get_weapon_ids(column='name')
        )
        df = df.reset_index()\
            .groupby('attacker_weapon_id').sum()\
            .sort_values('index', ascending=True)
        self._plot_bar(
            df['index'].values,
            df.index.values,
            'Weapon Usage - {}'.format(self._outfits_for_title(factions)),
            'Kills',
            0.2,
            color=self._get_weapon_ids('name', 'faction_id')
        )

    def plot_weapon_deaths(self, *factions):
        factions = self._verify_factions(factions)
        df = self._calc_weapon_stats(
            'attacker_weapon_id',
            'character_id',
            faction=factions,
            ids=self._get_weapon_ids(column='name')
        )
        df = df.reset_index()\
            .groupby('attacker_weapon_id').sum()\
            .sort_values('index', ascending=True)

        self._plot_bar(
            df['index'].values,
            df.index.values,
            'Death Causes - {}'.format(self._outfits_for_title(factions)),
            'Deaths',
            0.2,
            color=self._get_weapon_ids('name', 'faction_id')
        )

    def plot_players_heavy(self, *factions):
        factions = self._verify_factions(factions)
        self._plot_class_stats(
            factions,
            'Heavy Assault',
            'attacker_loadout_id',
            'attacker_character_id',
            'Heavy Assault Kills',
            'Kills'
        )

    def plot_players_medic(self, *factions):
        factions = self._verify_factions(factions)
        self._plot_class_stats(
            factions,
            'Medic',
            'attacker_loadout_id',
            'attacker_character_id',
            'Medic Kills',
            'Kills'
        )

    def plot_players_engineer(self, *factions):
        factions = self._verify_factions(factions)
        self._plot_class_stats(
            factions,
            'Engineer',
            'attacker_loadout_id',
            'attacker_character_id',
            'Engineer Kills',
            'Kills'
        )

    def plot_players_infiltrator(self, *factions):
        factions = self._verify_factions(factions)
        self._plot_class_stats(
            factions,
            'Infiltrator',
            'attacker_loadout_id',
            'attacker_character_id',
            'Infiltrator Kills',
            'Kills'
        )

    def plot_players_la(self, *factions):
        factions = self._verify_factions(factions)
        self._plot_class_stats(
            factions,
            'Light Assault',
            'attacker_loadout_id',
            'attacker_character_id',
            'Light Assault Kills',
            'Kills'
        )

    def plot_vehicle_kills(self, *factions, color='blue'):
        factions = self._verify_factions(factions)
        df = self._calc_weapon_stats('attacker_vehicle_id', 'attacker_character_id', factions, self._get_vehicle_ids(column='name'))
        df = df.reset_index().groupby('attacker_vehicle_id').sum().sort_values('index', ascending=True)
        self._plot_bar(
            df['index'].values,
            df.index.values,
            'Vehicle Usage - {}'.format(self._outfits_for_title(factions)),
            'Kills',
            0.23,
            color=color
        )

    # TODO: Fix method
    # def _plot_vehicle_deaths(self, *factions, color='blue'):
    #     factions = self._verify_factions(factions)
    #     df = self._calc_weapon_stats('attacker_vehicle_id', 'character_id', factions,
    #                                  self._get_vehicle_ids(column='name'))
    #     df = df.reset_index().groupby('attacker_vehicle_id').sum().sort_values('index', ascending=True)
    #     self._plot_bar(
    #         df['index'].values,
    #         df.index.values,
    #         'Vehicle Deaths - {}'.format(self._outfits_for_title(factions)),
    #         'Kills',
    #         0.23,
    #         color=color
    #     )

    def plot_players_vdestruction(self, *factions):
        factions = self._verify_factions(factions)
        self._plot_exp_stats(
            '*Vehicle%20Destruction',
            'count',
            'Total Amount of Vehicle Kills',
            'Vehicle Kills',
            y_offset=0.2,
            faction=factions
        )

    def plot_players_revives(self, *factions):
        factions = self._verify_factions(factions)
        self._plot_exp_stats(
            '*Revive',
            'count',
            'Total Amount of Revives',
            'Revives',
            y_offset=0.2,
            faction=factions
        )

    def plot_players_repairs(self, *factions):
        factions = self._verify_factions(factions)
        self._plot_exp_stats(
            '*Repair',
            'sum',
            'Total Amount of Repair XP',
            'Repair XP',
            y_offset=0.2,
            faction=factions
        )

    def plot_players_heals(self, *factions):
        factions = self._verify_factions(factions)
        self._plot_exp_stats(
            '*Heal',
            'sum',
            'Total Amount of Heal XP',
            'Heal XP',
            y_offset=0.2,
            faction=factions
        )

    def plot_players_spawns(self, *factions):
        factions = self._verify_factions(factions)
        self._plot_exp_stats(
            ['Squad%20Spawn', 'Spawn%20Bonus'],
            'count',
            'Total Amount of Spawn-Ins provided',
            'Spawn-Ins',
            y_offset=0.2,
            faction=factions
        )

    def player_stats(self, *factions, with_revives=True):
        factions = self._verify_factions(factions)
        df = self._player_stats(factions, with_revives=with_revives)
        return df

    ''' Private Plotting Methods '''

    def _plot_timeline(self, event_name:str or list, title, ylabel, column:str or list='character_id', agg_fun='count', sub_idx:list=None, div_idx:list=None, climit:int or list=100000, figsize=(14, 5)):
        if isinstance(event_name, list):
            vals = []
            for i in range(len(event_name)):
                vals.append(self._calc_timeline(event_name[i], column[i], agg_fun=agg_fun, climit=climit))

            if sub_idx is not None:
                vals[sub_idx[0]] = self._substract_timeline(vals[sub_idx[0]], vals[sub_idx[1]])
                del vals[sub_idx[1]]
            if div_idx is not None:
                vals[div_idx[0]] = self._divide_timeline(vals[div_idx[0]], vals[div_idx[1]])
                del vals[div_idx[1]]
            if len(vals) > 1:
                raise ValueError('sub_idx or div_idx incorrect.')
            vals = vals[0]
        else:
            vals = self._calc_timeline(event_name, column, agg_fun=agg_fun, climit=climit)

        vals = self._discretize_timeline(vals)

        plt.figure(figsize=figsize)
        for f in self._outfits.keys():
            plt.plot([v[1] for v in vals[f]], [v[0] for v in vals[f]], c=self._colors[f])

        plt.grid(True, alpha=0.2)
        plt.title(title)
        plt.xlabel('Elapsed Time [min]')
        plt.ylabel(ylabel)
        plt.show()

    def _plot_timeline_with_respawn(self, event_name:str or list, title, ylabel, column:str or list='character_id', agg_fun='count', sub_idx:list=None, div_idx:list=None, climit:int or list=100000, figsize=(14, 5)):
        if isinstance(event_name, list):
            vals = []
            for i in range(len(event_name)):
                vals.append(self._calc_timeline_with_respawn(column[i]))

            if sub_idx is not None:
                vals[sub_idx[0]] = self._substract_timeline(vals[sub_idx[0]], vals[sub_idx[1]])
                del vals[sub_idx[1]]
            if div_idx is not None:
                vals[div_idx[0]] = self._divide_timeline(vals[div_idx[0]], vals[div_idx[1]])
                del vals[div_idx[1]]
            if len(vals) > 1:
                raise ValueError('sub_idx or div_idx incorrect.')
            vals = vals[0]
        else:
            vals = self._calc_timeline_with_respawn(column)

        plt.figure(figsize=figsize)
        for f in self._outfits.keys():
            plt.plot([v[1] for v in vals[f]], [v[0] for v in vals[f]], c=self._colors[f])

        plt.grid(True, alpha=0.2)
        plt.title(title)
        plt.xlabel('Elapsed Time [min]')
        plt.ylabel(ylabel)
        plt.show()

    def _plot_hist(self, row_name, bins, figsize=(15, 10)):
        fig, axs = plt.subplots(len(self._outfits), 1, sharex=True, sharey=True, figsize=figsize)
        i = 0
        for fac, tag in self._outfits.items():
            data = self._player_stats([fac])
            axs[i].hist(data[row_name], color=self._colors.get(fac), bins=bins[i])
            axs[i].grid(True, alpha=0.2)
            axs[i].set_ylabel('Amount of Players')
            i += 1
        axs[0].set_title('Distribution of {}'.format(row_name))
        plt.show()

    # def _plot_weapon_stats(self, obj_column, char_column, faction='NC'):
    #     title = 'Weapons Usage - {}'.format(self._outfits.get(faction))
    #     xlabel = 'Deaths' if char_column == 'character_id' else 'Kills'
    #     df = self._calc_weapon_stats(obj_column, char_column, faction=faction, ids=self._get_weapon_ids())
    #     df = df.reset_index().groupby('attacker_weapon_id').sum().sort_values('index', ascending=True)
    #     self._plot_bar(
    #         df['index'].values,
    #         df.index.values,
    #         title,
    #         xlabel,
    #         0.2,
    #         color=self._get_weapon_ids(index='name', column='faction_id')
    #     )

    def _plot_class_stats(self, faction, class_name, loadout_column, char_column, title, xlabel, y_offset=0.2):
        if isinstance(faction, str): faction = [faction]
        df = self._calc_weapon_stats(loadout_column, char_column, faction=faction, ids=self._get_loadout_ids(column='code_name'))
        df_all = []
        for f in faction:
            df_all.append(df[df[loadout_column] == f + ' ' + class_name])
        df = pd.concat(df_all).sort_values('index')
        self._plot_bar(
            df['index'].values,
            df[char_column].values,
            title,
            xlabel,
            y_offset,
            color=self._get_players(faction, 'name', 'faction_id')
        )

    def _plot_exp_stats(self, url_name, agg_fun, title, xlabel, faction=('VS', 'NC', 'TR'), y_offset=0.2, climit=200, color='blue'):
        if isinstance(faction, str): faction = [faction]
        df = self._calc_exp_stats(url_name, agg_fun, faction=faction, climit=climit)
        self._plot_bar(
            df.amount.values,
            df.character_id.values,
            title,
            xlabel,
            y_offset=y_offset,
            color=self._get_players(faction, index='name', column='faction_id'))

    def _plot_bar(self, values, labels, title, xlabel, y_offset, figsize=None, color=None):
        if figsize is None:
            y_size = int(len(labels) * 0.3)
            figsize = (12, y_size)
        plt.figure(figsize=figsize)
        blist = plt.barh(labels, values)
        plt.ylim(-1, len(values))

        if isinstance(color, pd.Series):
            ids = color
            for i, l in enumerate(labels):
                f = ids.loc[l]
                if len(f) > 1:
                    f = ['0']
                blist[i].set_color(self._colors[self._factions[f[0]]])
        else:
            for b in blist:
                b.set_color(color)

        x_offset = max(values) * 0.005
        for i, v in enumerate(values):
            plt.text(v + x_offset, i - y_offset, str(int(v)))
        plt.yticks(ticks=labels, labels=labels)
        plt.xlabel(xlabel)
        plt.title(title)
        plt.grid(True, axis='x', alpha=0.3)
        plt.show()

    ''' Calculation Methods '''

    def _calc_timeline(self, event_name, column='character_id', climit=10000, agg_fun='count'):
        t_open, t_start, t_end = self._get_match_time()
        if event_name != 'Death':
            ids = self._get_exp_ids(event_name, args={'c:show':'experience_id,description'})
            data_filtered = self._filter(experience_id=ids.index)
        else:
            data_filtered = self._filter(event_name=event_name)

        vals = {}
        for f in self._outfits.keys():
            val = [[0, 0]]
            players = self._outfits_loaded.loc[self._outfits[f]].players
            for i, row in self._filter(data=data_filtered, args={column: players.index}).iterrows():
                if row.timestamp < t_start: continue
                val_old = val[-1][0]
                if agg_fun == 'sum':
                    val_new = val_old + int(row.amount)
                elif agg_fun == 'count':
                    val_new = val_old + 1
                else:
                    raise AttributeError(
                        'Invalid aggregation function \'{}\'. Use \'sum\' or \'count\'.'.format(agg_fun))
                val.append([val_new, (row.timestamp - t_start).total_seconds() / 60])
            vals[f] = val
        return vals

    def _calc_timeline_with_respawn(self, column):
        t_open, t_start, t_end = self._get_match_time()
        #column = 'attacker_character_id'
        t_respawn = pd.Timedelta('00:02:00')
        df_rev = self._filter(experience_id=self._get_exp_ids('*Revive').index)
        df = self._filter(event_name='Death')
        vals = {}
        for f in self._outfits.keys():
            val = [[0, 0]]
            players = self._outfits_loaded.loc[self._outfits[f]].players
            for i, row in self._filter(data=df, args={column: players.index}).iterrows():
                if row.timestamp < t_start: continue
                df_respawn = df_rev[(df_rev.timestamp >= row.timestamp) & (df_rev.timestamp < row.timestamp+t_respawn)]
                val_new = val[-1][0]
                if row.character_id not in df_respawn.other_id:
                    val_new = val_new + 1
                    val.append([val_new, (row.timestamp - t_start).total_seconds() / 60])
            vals[f] = val
        return vals

    def _substract_timeline(self, val1, val2):
        subs = {}
        for f in self._outfits.keys():
            sub = [[0, 0]]
            v1 = np.array(val1[f])
            v2 = np.array(val2[f])
            v1_idx = 0
            v2_idx = 0
            while True:
                if v1_idx + 1 == len(v1): break
                if v2_idx + 1 == len(v2): break

                t_v1_next = v1[v1_idx + 1, 1]
                t_v2_next = v2[v2_idx + 1, 1]
                if t_v1_next < t_v2_next:
                    v1_idx += 1
                    sub.append([sub[-1][0]+1, t_v1_next])
                elif t_v1_next > t_v2_next:
                    v2_idx += 1
                    sub.append([sub[-1][0] - 1, t_v2_next])
                else:
                    v1_idx += 1
                    v2_idx += 1
            subs[f] = sub
        return subs

    def _divide_timeline(self, val1, val2):
        fracs = {}
        for f in self._outfits.keys():
            frac = [[0, 0]]
            v1 = np.array(val1[f])
            v2 = np.array(val2[f])
            t_cur = 0
            k_idx = 0
            d_idx = 0
            while True:
                if k_idx + 1 == len(v1): break
                if d_idx + 1 == len(v2): break
                t_k = v1[k_idx + 1, 1]
                t_d = v2[d_idx + 1, 1]
                if t_k < t_d:
                    k_idx += 1
                    t_cur = t_k
                elif t_k > t_d:
                    d_idx += 1
                    t_cur = t_d
                else:
                    k_idx += 1
                    d_idx += 1
                    t_cur = t_k
                if v2[d_idx, 0] == 0:
                    frac.append([v1[k_idx, 0], t_cur])
                else:
                    frac.append([v1[k_idx, 0] / v2[d_idx, 0], t_cur])
            fracs[f] = frac
        return fracs

    def _discretize_timeline(self, vals):
        vals_dis = {}
        for f in self._outfits.keys():
            val = vals[f]
            val_dis = []
            for i in range(len(val)-1):
                val_dis.append(val[i])
                val_dis.append([val[i][0], val[i+1][1]])
            val_dis.append(val[-1])
            vals_dis[f] = val_dis
        return vals_dis

    def _calc_weapon_stats(self, wpn_column, char_column, faction=('VS', 'NC', 'TR'), ids: pd.Series = None):
        players = self._players_from_faction(faction)
        df = self._filter(event_name='Death', args={wpn_column: ids.index, char_column: players.index})
        df[wpn_column] = df[wpn_column].replace(ids.to_dict())
        df[char_column] = df[char_column].replace(players.to_dict()['name'])
        df = df.groupby([char_column, wpn_column]).count().sort_values('index', ascending=True).reset_index()
        return df

    def _calc_exp_stats(self, url_name, agg_fun, char_column='character_id', faction=('VS', 'NC', 'TR'), climit=10000):
        # Getting Players
        players = self._players_from_faction(faction)

        # Transforming Data
        ids = self._get_exp_ids(url_name, climit=climit, process=False)
        df = self._filter(
            event_name='GainExperience',
            args={'character_id': players.index, 'experience_id': ids.set_index('experience_id').index}
        )
        df[char_column] = df[char_column].replace(players.to_dict()['name'])
        # Grouping
        if agg_fun == 'count':
            df = df[[char_column, 'amount']].groupby(char_column).count().sort_values('amount', ascending=True).reset_index()
        elif agg_fun == 'sum':
            df['amount'] = df['amount'].astype('float64')
            df = df[[char_column, 'amount']].groupby(char_column).sum().sort_values('amount', ascending=True).reset_index()
        else:
            raise AttributeError('Invalid aggregation function \'{}\'. Use \'sum\' or \'count\'.'.format(agg_fun))

        return df

    def _player_stats(self, factions, with_revives=True):
        factions = self._verify_factions(factions)
        df = pd.DataFrame()
        for f in factions:
            players = self._outfits_loaded.loc[self._outfits[f]].players
            data_kills = self._filter(event_name='Death', attacker_character_id=players.index)
            data_kills['attacker_character_id'] = data_kills['attacker_character_id'].replace(players.to_dict()['name'])

            data_deaths = self._filter(event_name='Death', character_id=players.index)
            data_deaths['character_id'] = data_deaths['character_id'].replace(players.to_dict()['name'])

            kills = data_kills.groupby(by=['attacker_character_id']).count()['character_id']
            headshots = data_kills[data_kills['is_headshot'] == '1']\
                .groupby(by=['attacker_character_id']).count()['timestamp']
            deaths = data_deaths.groupby(by=['character_id']).count()['event_name']
            data_players = pd.concat([kills, deaths, headshots], axis=1)\
                .rename(columns={"character_id": "Kills", "event_name": "Deaths", "timestamp": "Headshots"})\
                .replace(np.nan, 0)\
                .astype('int')
            df = pd.concat([df, data_players])
        if with_revives:
            df_rev = self._calc_exp_stats('*Revive', 'count', char_column='other_id', faction=factions)
            df['Deaths'] = df['Deaths'].sub(df_rev.set_index('other_id').astype('int64')['amount']).fillna(df['Deaths'])
        # Add KDR
        df['KDR'] = (df['Kills'] / df['Deaths'])\
            .replace(np.inf, 0)\
            .apply(round, ndigits=2)
        # Add HSR
        df['HSR'] = (df['Headshots'] / df['Kills'] * 100)\
            .replace(np.inf, 0)\
            .replace(np.nan, 0)\
            .apply(round)
        return df

    ''' Utility Methods '''

    def _get_match_time(self):
        data_captures = self._filter(event_name='FacilityControl')
        t_open = data_captures.timestamp.min()
        t_start = t_open + pd.Timedelta('00:20:00')
        t_end = data_captures.timestamp.max()
        return t_open, t_start, t_end

    def _filter(self, data=None, args={}, **conditions):
        zone_id = {}
        if self._zone_id is not None:
            zone_id = {'zone_id': self._zone_id}
        conditions = {**zone_id, **conditions, **args}

        # check is data is provided
        if data is None:
            df = self._data
        else:
            df = data

        # apply filter conditions
        for cond in conditions.items():
            if isinstance(cond[1], pd.Index):
                df = df[df[cond[0]].isin(cond[1])]
            else:
                df = df[df[cond[0]] == str(cond[1])]
        return df.copy()

    def _players_from_faction(self, faction):
        if isinstance(faction, str): faction = [faction]
        players = []
        for f in faction:
            players.append(self._outfits_loaded.loc[self._outfits[f]].players)
        return pd.concat(players)

    def _outfits_for_title(self, faction):
        if isinstance(faction, str): faction = [faction]
        outfit_name = ''
        for i, f in enumerate(faction):
            outfit_name += self._outfits[f]
            if i < len(faction) - 1:
                outfit_name += ', '
        return outfit_name

    def _data_check(self):
        if self._zone_id is None:
            raise ValueError('No match has been selected! Use \'set_match(...)\' to select one.')

    def _verify_factions(self, factions):
        if len(factions) == 0:
            factions = ('VS', 'NC', 'TR')
        f_vaild = list(self._outfits.keys())
        correct = []
        wrong = []
        for f in factions:
            if not isinstance(f, str):
                raise TypeError('Input has to be a String. Use \'value\' or \"value\"')
            if f not in f_vaild:
                wrong.append(f)
            else:
                correct.append(f)
        if len(wrong) > 0:
            raise KeyError('Invalid input(s) {}. Valid inputs are: {}'.format(wrong, f_vaild))
        return correct

    ''' Data Loading Methods '''

    def _from_json(self, files, zone_id=None):
        zone_ids = self._from_census('zone', args={'c:show': 'zone_id,code'})
        if not isinstance(files, list): files = [files]

        data = pd.DataFrame()
        df = None
        for f in files:
            print('Loading File \'{}\' ... '.format(f), end='')
            max_tries = 500
            for i in range(max_tries):
                try:
                    df = pd.read_json(f, orient='list', dtype=False)
                    break
                except ValueError:
                    if i == max_tries-1:
                        raise LookupError('Failed to load file \'{}\''.format(f))
            try:
                df = df.payload.apply(pd.Series)
                df = df[(df.zone_id.isin(zone_ids.index) == False) & (pd.isna(df.zone_id) == False)]
                df['timestamp'] = df.timestamp.apply(pd.to_datetime, unit='s')
                df = df.reset_index()
            except AttributeError:
                pass
            if zone_id is not None:
                df = df[df.zone_id == zone_id]
            data = pd.concat([data, df], axis=0)
            print('done')
        print('Loading Finished')
        self._data = data
        if len(self.zone_ids) == 1:
            self.set_match(self.zone_ids[0])

    def _from_census(self, id_category: str, climit=10000, process=True, args={}, **querys):
        querys = {**querys, **args}

        # create the string query body
        multiples = []
        body = ''
        for q in querys.items():
            value = q[1]
            if isinstance(q[1], list):
                if len(multiples) == 0:
                    multiples = q[1]
                else:
                    raise AttributeError('Only one query element is allowed to have multiple values!')
                value = '{}'
            body += '{}={}&'.format(q[0], value)
        if len(multiples) == 0: multiples = [0]

        # request all queries from census
        ids = pd.DataFrame()
        n_max = 3
        ids_new = None
        column_name = None
        for i, m in enumerate(multiples):
            for r in range(n_max):
                try:
                    ids_new = pd.DataFrame(pd.read_json(self._census_url.format(
                        id_category,
                        body.format(m),
                        climit
                    ), typ='series')[0])
                    #print('requesting..', id_category, body.format(m))
                    break
                except (ConnectionResetError, ValueError):
                    if i == n_max:
                        raise ConnectionResetError('Unable to load data from Census')
                    t_wait = 60
                    for t in range(t_wait):
                        end = '\r'
                        if t_wait-t+1 == 0:
                            end = ''
                        print('Census request limit reached. Waiting for {}s..'.format(t_wait-t+1), end=end)

                        time.sleep(1)
                    #print('\n')
            if process:
                ids_new[ids_new.columns[0]] = ids_new[ids_new.columns[0]]
                ids_new = ids_new.set_index(ids_new.columns[0])

            ids = pd.concat([ids, ids_new])
        if process:
            ids = ids.dropna()
            if isinstance(ids[ids.columns[0]].iloc[0], dict):
                ids[ids.columns[0]] = ids[ids.columns[0]].apply(lambda d: d['en'])
        return ids

    def _load_outfit(self, **query_args):
        factions = ['VS', 'NC', 'TR']
        val = list(query_args.values())[0]
        if val in self._outfits_loaded.outfit_id.values:
            row = self._outfits_loaded[self._outfits_loaded.outfit_id == val]
            return row.name
        elif val in self._outfits_loaded.index:
            row = self._outfits_loaded.loc[val]
            return row.name
        else:
            df = self._from_census(
                'outfit',
                args={**query_args, 'c:resolve': 'member_character'},
                process=False
            )
            if len(df) == 0:
                raise KeyError('No outfit exists with {} {}.'.format(list(query_args.keys())[0], list(query_args.values())[0]))
            tag = df.alias[0]
            outfit_id = df.outfit_id.loc[0]
            self._data['outfit_id'] = self._data['outfit_id'].replace(outfit_id, tag)
            df = pd.DataFrame(df.members[0])
            faction_id = int(df.faction_id.iloc[0])
            df['name'] = df['name'].apply(pd.Series)['first']
            df = df[['name', 'character_id', 'faction_id']].set_index('character_id')
            row = pd.Series({
                'outfit_id': outfit_id,
                'faction': factions[faction_id-1],
                'players': df
            })
            row.name = tag
            self._outfits_loaded = self._outfits_loaded.append(row)
            return tag

    def _get_players(self, faction: list, index=None, column=None):
        df = []
        for f in faction:
            df.append(self._outfits_loaded.loc[self._outfits[f]].players)
        df = pd.concat(df)
        if index is not None:
            df = df.reset_index().set_index(index)
        if column is not None:
            df = df[column]
        return df

    def _get_loadout_ids(self, index=None, column=None):
        if self._loadouts is None:
            self._loadouts = self._from_census('loadout', args={'c:show': 'loadout_id,code_name,faction_id'})
        df = self._loadouts.copy()
        if index is not None:
            df = df.reset_index().set_index(index)
        if column is not None:
            df = df[column]
        return df

    def _get_weapon_ids(self, index=None, column=None):
        if self._weapons is None:
            self._weapons = self._from_census('item', args={'c:show': 'item_id,name.en,faction_id'})
        df = self._weapons.copy()
        if index is not None:
            df = df.reset_index().set_index(index)
        if column is not None:
            df = df[column]
        return df

    def _get_vehicle_ids(self, index=None, column=None):
        if self._vehicles is None:
            df = self._from_census(
                'vehicle',
                args={
                    'c:show': 'vehicle_id,name.en',
                    'c:join': 'vehicle_faction^show:faction_id^inject_at:faction_id^list:1'
                })
            # TODO: Extract faction ids for coloring
            self._vehicles = df
        df = self._vehicles.copy()
        if index is not None:
            df = df.reset_index().set_index(index)
        if column is not None:
            df = df[column]
        return df

    def _get_exp_ids(self, url_name, args={}, climit=1000, process=True):
        if url_name in self._exp_ids.keys():
            ids = self._exp_ids[url_name]
        else:
            ids = self._from_census('experience', description=url_name, args=args, process=process, climit=climit)
            self._exp_ids[url_name] = ids
        return ids

    @property
    def event_names(self):
        return pd.unique(self._data.event_name)

    @property
    def available_themes(self):
        return self._themes

    @property
    def zone_ids(self):
        return self._data.zone_id.unique()

    @property
    def data(self):
        return self._data
