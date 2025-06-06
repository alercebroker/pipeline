import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import Time
from datetime import date, datetime, timedelta
import json
import sqlalchemy as sa
import requests
from alerce.core import Alerce
import os, sys

stamp_name = 'stamp_classifier_2025_beta'
classifier_names = ['stamp_classifier',
                    stamp_name]
classifier_names2 = ['stamp_old',
                     'stamp_new']
classifier_versions = ['stamp_classifier_1.0.4',
                       'beta']

def find_cl_map(cl_list):
    cl_forsort = np.arange(len(cl_list))
    cl_map = {key: val for key, val in zip(cl_list, cl_forsort)}
    cl_map = pd.DataFrame([cl_map]).T
    
    return cl_map

def fix_ts_classname(df=None, col_class='class'):
    df[col_class + '_new'] = [cls_map_fromlowercase[x] for x in df[col_class]]
    df.drop(columns=[col_class], inplace=True)
    df.rename(columns={col_class + '_new': col_class}, inplace=True)

    return df

def sort_vcounts_cl(df=None, cl_map=None):
    df = pd.DataFrame(df)
    df['index'] = [cl_map.loc[x, 0] for x in df.index]
    df.sort_values(by=['index'], inplace=True)
    df.drop(columns=['index'], inplace=True)
    
    return df

def show_value_counts(objs=None, col=None, title=None, namefig=None,
                      figsize=None, show_list=False, create_fig=False,
                      fontsize=12):
    print(str(len(objs)) + ' objects')
    
    cls = pd.unique(objs[col])
    #print('' + str(len(cls)) + ' class(es)')

    if show_list:
        pd.set_option('display.max_rows', None)
        display(objs[col].value_counts())
        pd.set_option('display.max_rows', 30)
    
    if create_fig:
        df_hist = objs[col].value_counts()
        df_hist = df_hist.sort_values(ascending=True).copy()
        
        if figsize is None:
            fig, ax = plt.subplots(figsize=(4, len(cls) * 0.4))
        else:
            fig, ax = plt.subplots(figsize=(figsize[0], figsize[1]))
        
        df_hist.plot(ax=ax, kind='barh', xlabel='Count',
                     #ylabel='Class name',
                     ylabel='',
                     logx=True,# figsize=(14, 5),
                     alpha=0.7, fontsize=fontsize)
        
        plt.title(title, fontsize=fontsize + 1)
        
        df_hist = pd.DataFrame(df_hist)
        
        xlims = ax.get_xlim()
        #df_hist['xpos'] = (xlims[1] - xlims[0]) * 0.995
        df_hist['xpos'] = (xlims[1] - xlims[0]) * 1.01
        
        #display(df_hist)
        
        for i, row in df_hist.iterrows():
            ax.annotate(row['count'].astype(int), (row['xpos'],
                                               df_hist.index.get_loc(i)),
                        #rotation=90,
                        fontsize=fontsize - 1, ha='left', va='center')
                        #fontsize=10, ha='right', va='center')
        
        if namefig is not None:
            plt.tight_layout()
            fig.savefig(namefig)#, bbox_inches='tight')
        
        #objs[col].value_counts().plot(kind='bar', title=title,
        #                              figsize=(12, 4))
        #plt.show()

    return

def plot_scatter(df=None, title=None, propx=None, propy=None,
                 propcolor=None, fontsize=None, alpha=0.7,
                 namefig=None):
    if fontsize is not None:
        plt.rcParams.update({'font.size': fontsize})
    
    cmap = 'coolwarm'
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    sc = ax.scatter(df[propx], df[propy], c=df[propcolor],
                    cmap=cmap, s=0.5, alpha=alpha)
    plt.colorbar(sc, ax=ax, label=propcolor)
    ax.set_xlabel(propx)
    ax.set_ylabel(propy)
    
    ax.set_title(title)
    
    if namefig is not None:
        plt.tight_layout()
        fig.savefig(namefig)
    
    if fontsize is not None:
        plt.rcParams.update({'font.size': 12})
    return

def plot_cols_hist(df=None, cols=None, bins=None, xlims=None,
                   title=None, fontsize=None, namefig=None, showfig=True):
    alpha = 0.7
    
    if fontsize is not None:
        plt.rcParams.update({'font.size': fontsize})
    
    for col in cols:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df[col], bins=bins, range=xlims, alpha=alpha)
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.set_yscale('log')
        ax.set_title(title)
        
        if namefig is not None:
            plt.tight_layout()
            fig.savefig(namefig)
    
    if fontsize is not None:
        plt.rcParams.update({'font.size': 12})
    
    if not showfig:
        plt.close(fig)

    return

def plot_2distribs_cls(df=None, labels_pred_map=None, firstmjd_min='',
                       fontsize=12, namefig=None, showfig=True):
    plt.rcParams.update({'font.size': fontsize})

    nobjs = len(df)

    fig, ax = plt.subplots(ncols=2)#, figsize=(16, 4))

    for i, (clf_name, clf_version) in enumerate(zip(classifier_names2,
                                                    classifier_versions)):
        col_pred = 'class_name_' + clf_name
        mask = df[col_pred].notna()
        df_aux = df[mask].copy()

        title = clf_name + ' (' + str(len(df_aux)) + '/' + str(nobjs) \
                + ' objs), firstmjd=' + str(int(firstmjd_min))
        df_hist = df_aux[col_pred].value_counts()
        aux = labels_pred_map[~labels_pred_map.index.isin(df_hist.index)].copy()
        
        if len(aux) > 0:
            aux.loc[aux.index, 0] = 0
            aux.rename(columns={0: 'count'}, inplace=True)
            df_hist = pd.concat([df_hist, aux], axis=0)
        
        df_hist = sort_vcounts_cl(df=df_hist, cl_map=labels_pred_map)
        df_hist.plot(ax=ax[i], kind='barh', logx=True,
                    xlabel='Frequency', ylabel='Predicted class',
                    alpha=0.7, fontsize=fontsize,
                    figsize=(12, 4), legend=False)
        ax[i].set_title(title, fontsize=fontsize + 1)

        df_hist = pd.DataFrame(df_hist)
        xlims = ax[i].get_xlim()
        #df_hist['xpos'] = (xlims[1] - xlims[0]) * 0.995
        df_hist['xpos'] = (xlims[1] - xlims[0]) * 1.01
        for j, row in df_hist.iterrows():
            ax[i].annotate(row['count'].astype(int), (row['xpos'],
                                                      df_hist.index.get_loc(j)),
                           fontsize=fontsize - 1, ha='left', va='center')

    plt.tight_layout()
    
    if namefig is not None:
        fig.savefig(namefig, bbox_inches='tight')
    
    if fontsize is not None:
        plt.rcParams.update({'font.size': 12})
    
    if not showfig:
        plt.close(fig)

    return

def plot_2distribs_radecprob(df=None, cls=None, firstmjd_min='',
                             fontsize=12, namefig=None, showfig=True,gal_plane_ra=None
                             ,gal_plane_dec=None,ecl_plane_ra=None, ecl_plane_dec=None):
    plt.rcParams.update({'font.size': fontsize})
    
    cmap = 'coolwarm'
    alpha = 0.5

    for (propx, propy, coordtype) in zip(['meanra'],# 'gal_l', 'ecl_lon'],
                                         ['meandec'],# 'gal_b', 'ecl_lat'],
                                         ['equatorial']):#, 'galactic', 'ecliptic']):
        #print(coordtype)
        fig, ax = plt.subplots(ncols=len(cls), nrows=2, figsize=(len(cls) * 5, 8))
        suptitle = classifier_names2[0] + ' (top), '+ classifier_names2[1] \
                   + ' (bottom), firstmjd=' + str(int(firstmjd_min))
        fig.suptitle(suptitle,
                     fontsize=fontsize + 2)
        
        for i, clf_name in enumerate(classifier_names2):
            propcolor = 'probability_' + clf_name
            
            for j, cl_this in enumerate(cls):
                if cl_this == 'all':
                    df_aux = df[df['class_name_' + clf_name].notna()].copy()
                else:
                    mask = df['class_name_' + clf_name] == cl_this
                    df_aux = df[mask].copy()
    
                if len(df_aux) > 0:
                    #title_aux = cl_this + ', ' + clf_name + ' (' + str(len(df_aux)) \
                    #            + ' objs), mjd=' + str(int(firstmjd_min))
                    title_aux = cl_this + ' (' + str(len(df_aux)) + ' objs)'
                    
                    sc = ax[i][j].scatter(df_aux[propx], df_aux[propy], c=df_aux[propcolor],
                                       cmap=cmap, s=10, alpha=alpha)
                    plt.colorbar(sc, ax=ax[i][j], label=propcolor)

                    if coordtype == 'equatorial':
                        ax[i][j].scatter(gal_plane_ra, gal_plane_dec, s=5, c='magenta', alpha=0.5)
                        ax[i][j].scatter(ecl_plane_ra, ecl_plane_dec, s=5, c='cyan', alpha=0.5)
                    
                    ax[i][j].set_xlabel(propx)
                    ax[i][j].set_ylabel(propy)
                    ax[i][j].set_title(title_aux, fontsize=fontsize + 1)
                    ax[i][j].set_xlim([0, 360])
                    ax[i][j].set_ylim([-90, 90])

    plt.tight_layout()
    
    if namefig is not None:
        fig.savefig(namefig, bbox_inches='tight')
    
    if fontsize is not None:
        plt.rcParams.update({'font.size': 12})
    
    if not showfig:
        plt.close(fig)
    return

def print_link(index=None, classifier=None):
    expr1 = 'https://alerce.online/?oid='
    expr2 = '&oid='.join(list(index))
    expr3 = '&selectedClassifier=' + classifier + '&page=1'
    expr = expr1 + expr2 + expr3

    display(HTML("<a href='%s' target=\"_blank\"> %s <a>" % (expr, expr)))
    return

def find_objs_perday(firstmjd_min=None, conn=None):
    query = '''
    SELECT
        oid, ndet, meanra, meandec, deltajd,
        firstmjd, lastmjd, step_id_corr
    FROM
        object
    WHERE
        firstmjd > %s
        AND firstmjd <= %s
    ORDER BY lastmjd DESC
    ''' % (str(firstmjd_min),
           str(firstmjd_min + 1))
    #print(query)
    
    df = pd.read_sql_query(query, conn)
    if len(df) > 0:
        df.set_index('oid', inplace=True)
    #print(len(df))
    #display(df)

    return df

def find_probs_r1(oids=None, clf_name=None, conn=None):
    query = '''
    SELECT
        oid, class_name, classifier_name, classifier_version, probability
    FROM
        probability
    WHERE
        classifier_name = '%s'
        AND ranking = 1
        AND oid IN (%s)
    ''' % (clf_name,
           ','.join(["'%s'" % oid for oid in oids]))
    #print(query)
    
    df = pd.read_sql_query(query, conn)
    if len(df) > 0:
        df.set_index('oid', inplace=True)
        df.sort_values(by='probability', ascending=False, inplace=True)
    #print(len(df))
    display(df)

    return df
