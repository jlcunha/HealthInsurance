class HealthInsuranceFunctions( object ):
    '''This HealthInsuranceFunctions(class) was made to save the functions used in health insurance cross-sell project.'''
    
    # Stacked bar to Data Visualization
    def stacked_bar (index, columns, figsize, xlabel, text_x_location, text_y_location):
        cross_tab_prop = pd.crosstab( index = index,
                                 columns = columns,
                                 normalize='index')

        cross_tab =  pd.crosstab( index = index,
                                 columns = columns)

        cross_tab_prop.plot(kind='bar',
                            stacked='true',
                            colormap='tab10',
                            figsize=figsize)

        plt.legend(loc='upper left', ncol=2)
        plt.xlabel(xlabel)
        plt.ylabel('Proportion')

        for n, x in enumerate([*cross_tab.index.values]):
            for (proportion, count, y_loc) in zip(cross_tab_prop.loc[x],
                                                  cross_tab.loc[x],
                                                  cross_tab_prop.loc[x].cumsum()):

                plt.text(x=n - text_x_location,
                         y=(y_loc - proportion) + (proportion / text_y_location),
                         s=f'{count}\n({np.round(proportion * 100, 1)}%)', 
                         color="black",
                         fontsize=12,
                         fontweight="bold")
        plt.show()

    ###################################################################################################################
    # Function to calculate the precision 
    def precision_at_k( data, k=2000 ):
        # reset index
        data = data.reset_index( drop=True )

        # create ranking order
        data['ranking'] = data.index + 1

        data['precision_at_k'] = data['response'].cumsum() / data['ranking']

        return data.loc[k, 'precision_at_k']

    ###################################################################################################################
    # Function to calculate the recall
    def recall_at_k( data, k=2000 ):
        # reset index
        data = data.reset_index( drop=True )

        # create ranking order
        data['ranking'] = data.index + 1

        data['recall_at_k'] = data['response'].cumsum() / data['response'].sum()

        return data.loc[k, 'recall_at_k']

    ###################################################################################################################
    # Function to create the data frame with the original response and the predicted values
    def make_score( X_val, y_val, yhat_model):
        # copy data
        data = X_val.copy()
        data['response'] = y_val.copy()

        # propensity score
        data['score'] = yhat_model[:,1].tolist()

        # sorted clients by propensity score
        data = data.sort_values('score', ascending=False)

        return data

    ###################################################################################################################
    # Function to create the data frame with the profits per samples
    def profit_dataframe(data, insurance_price, deal_cost):
        x_nsamples = []
        y_profit = []
        y_baseline = []

        number_of_positives = data[data['response'] == 1].shape[0]
        population_size = data.shape[0]

        for i in range(99):
            k_number = int(x_val.shape[0]*(i+1)/100)
            k_percent = (i+1)/100
            recall_at_kk = recall_at_k( data, k=k_number)
            profit = (recall_at_kk * number_of_positives * insurance_price) - (k_percent * population_size * deal_cost)
            baseline = (k_percent * number_of_positives * insurance_price) - (k_percent * population_size * deal_cost)

            x_nsamples.append(k_percent)
            y_profit.append(profit)
            y_baseline.append(baseline)

        df_profit = pd.DataFrame(columns = ['x_nsamples', 'y_profit', 'y_baseline'])
        df_profit['x_nsamples'] = x_nsamples
        df_profit['y_profit'] = np.round(y_profit,2)
        df_profit['y_baseline'] = np.round(y_baseline,2)

        return df_profit

    ###################################################################################################################
    # Function to made a line plot profits
    def line_profit(df_profit):
        plt.figure(figsize=(12,6))

        # getting the values 
        index = df_profit[df_profit['y_profit'] == df_profit['y_profit'].max()].index[0]
        y_value = int(df_profit[df_profit['y_profit'] == df_profit['y_profit'].max()]._get_value(index,'y_profit'))
        x_value = df_profit[df_profit['y_profit'] == df_profit['y_profit'].max()]._get_value(index,'x_nsamples')
        y_min_value = int(df_profit['y_profit'].min()*1.1)

        y_baseline = int(df_profit[df_profit['y_profit'] == df_profit['y_profit'].max()]._get_value(index,'y_baseline'))

        # text
        text = f'Percentage of sample = {x_value}\n Profit = {y_value}'

        text_baseline = f'Percentage of sample = {x_value}\n Profit = {y_baseline}'

        # baseline plot
        ax1 = sns.lineplot(x='x_nsamples', y='y_baseline', data=df_profit, marker='o', fillstyle='none', markeredgewidth=1, 
                                   markeredgecolor='black', markevery=[(index)], label='Baseline')

        ax1.text(x_value,(y_baseline + (((0.2*y_baseline)**2)**(1/2)) ), text_baseline, horizontalalignment='center', 
                 verticalalignment='center', fontsize=10);

        # model plot
        line_profit = sns.lineplot(x='x_nsamples', y='y_profit', data=df_profit, marker='o', fillstyle='none', markeredgewidth=1, 
                                   markeredgecolor='black', markevery=[(index)], ax= ax1, label='With Model')
        line_profit.set_xlabel('Percentage of sample')
        line_profit.set_ylabel('Profit')
        line_profit.set_title('Best K for the Biggest Profit');
        line_profit.set_ylim([y_min_value,y_value*1.5])

        # setting the plot information
        line_profit.text(x_value,(y_value * 1.2 ), text, horizontalalignment='center', verticalalignment='center', fontsize=10);

        return line_profit.plot

    ###################################################################################################################
    # Function to made a bar plot profit
    def profit_bar(df_base, insurance_price, deal_cost):
        population_size = df_base.shape[0]

        df_profit = profit_dataframe(data=df_base, insurance_price=insurance_price, deal_cost=deal_cost)

        # best profit
        index = df_profit[df_profit['y_profit'] == df_profit['y_profit'].max()].index[0]
        y_value = int(df_profit[df_profit['y_profit'] == df_profit['y_profit'].max()]._get_value(index,'y_profit'))

        # all base profit
        population_size = df_base.shape[0]
        interested_customers = df_base[df_base['response'] == 1].shape[0]
        revenue = interested_customers * insurance_price
        cost = population_size * deal_cost

        profit = revenue - cost

        plt.figure(figsize=(12,6))
        bar_plot = sns.barplot(x=['Selected Customers', 'All Customers'], y=[y_value, profit])
        bar_plot.set_title('Profit: Selected Customers X All Customers')
        bar_plot.set_ylabel('Profit')

        values = [y_value, profit]
        percentage_values = [np.round((y_value/profit*100),2), (profit/profit*100)]

        for rect, perc, vals  in zip(bar_plot.patches, percentage_values, values):    
            bar_plot.annotate(f"{perc}% ({vals})", (rect.get_x() + rect.get_width() / 2., rect.get_height()),
                         ha='center', va='center', fontsize=10, color='black', xytext=(0, 10),
                         textcoords='offset points')

        plt.tight_layout()

        return bar_plot.plot;

    ###################################################################################################################
    # Function to training the model for Cross Validation
    def training(train, test, fold_no, sample_size):
        x_train = train.drop(['response'],axis=1)
        y_train = train.response
        x_test = test.drop(['response'],axis=1)
        y_test = test.response
        model.fit(x_train, y_train)
        score = model.predict_proba( x_test )
        score_data = make_score(x_test, y_test, score)
        k = int(score_data.shape[0]*sample_size)
        precision_at_n_k = precision_at_k( score_data, k=k )
        recall_at_n_k = recall_at_k( score_data, k=k )

        return list([precision_at_n_k, recall_at_n_k])

    ###################################################################################################################
    # Function to run Cross Validation Stratifield
    def stratifield_cross_validation(x_data, y_data, dataset, skf, model, sample_size):
        cv_dict = {}

        fold_no = 1
        for train_index,test_index in skf.split(x_data, y_data):
            train = dataset.iloc[train_index,:]
            test = dataset.iloc[test_index,:]
            metrics_list = training(train, test, fold_no, sample_size)
            cv_dict[fold_no] = metrics_list
            print(f"fold :{fold_no} | precision {metrics_list[0]} | recall {metrics_list[1]}")

            fold_no += 1

        return(cv_dict)