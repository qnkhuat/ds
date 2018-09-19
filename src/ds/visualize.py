import matplotlib.pyplot as plt
import seaborn as sns

def corr_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    plt.show()


def confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    plt.show()

def feature_selection(alg,x,y):
    """
    Args:
       alg : alogortihm 
       x (dataframe)
       y (dataframe)
    """
    #base model
    print('BEFORE DT RFE Training Shape Old: ', x.shape) 
    print('BEFORE DT RFE Training Columns Old: ', x.columns.values)

    print("BEFORE DT RFE Training w/bin score mean: {:.2f}".format(
        base_results['train_score'].mean()*100)) 
    print("BEFORE DT RFE Test w/bin score mean: {:.2f}".format(
        base_results['test_score'].mean()*100))
    print("BEFORE DT RFE Test w/bin score 3*std: +/- {:.2f}".format(
        base_results['test_score'].std()*100*3))
    print('-'*10)



    #feature selection
    dtree_rfe = feature_selection.RFECV(dtree, step = 1, scoring = 'accuracy', cv = cv_split)
    dtree_rfe.fit(x,y)

    #transform x&y to reduced features and fit new model
    #alternative: can use pipeline to reduce fit and transform steps: http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
    X_rfe = x.columns.values[dtree_rfe.get_support()]
    rfe_results = model_selection.cross_validate(dtree, x[X_rfe], y, cv  = cv_split)

    #print(dtree_rfe.grid_scores_)
    print('AFTER DT RFE Training Shape New: ', x[X_rfe].shape) 
    print('AFTER DT RFE Training Columns New: ', X_rfe)

    print("AFTER DT RFE Training w/bin score mean: {:.2f}".format(
        rfe_results['train_score'].mean()*100)) 
    print("AFTER DT RFE Test w/bin score mean: {:.2f}".format(
        rfe_results['test_score'].mean()*100))
    print("AFTER DT RFE Test w/bin score 3*std: +/- {:.2f}".format(
        rfe_results['test_score'].std()*100*3))
    print('-'*10)


