import os
import sys
import django
from django.conf import settings
from django.core.management import execute_from_command_line

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 1. DJANGO SETTINGS
# ==========================================
if not settings.configured:
    settings.configure(
        DEBUG=True,
        SECRET_KEY='super-secret-development-key',
        ROOT_URLCONF=__name__,
        LOGIN_URL='/login/',
        ALLOWED_HOSTS=['*'],
        INSTALLED_APPS=[
            'django.contrib.admin',
            'django.contrib.auth',
            'django.contrib.contenttypes',
            'django.contrib.sessions',
            'django.contrib.messages',
            '__main__', # Treats this main.py file as a Django app!
        ],
        DATABASES={
            'default': {
                'ENGINE': 'django.db.backends.sqlite3',
                'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
            }
        },
        TEMPLATES=[{
            'BACKEND': 'django.template.backends.django.DjangoTemplates',
            'DIRS': [os.path.join(BASE_DIR, 'templates')],
            'APP_DIRS': True,
            'OPTIONS': {
                'context_processors': [
                    'django.template.context_processors.request',
                    'django.contrib.auth.context_processors.auth',
                    'django.contrib.messages.context_processors.messages',
                ],
            },
        }],
        MIDDLEWARE=[
            'django.middleware.security.SecurityMiddleware',
            'django.contrib.sessions.middleware.SessionMiddleware',
            'django.middleware.common.CommonMiddleware',
            'django.middleware.csrf.CsrfViewMiddleware',
            'django.contrib.auth.middleware.AuthenticationMiddleware',
            'django.contrib.messages.middleware.MessageMiddleware',
            'django.middleware.clickjacking.XFrameOptionsMiddleware',
        ]
    )

django.setup()

# ==========================================
# 2. MODELS (Data Architecture)
# ==========================================
from django.db import models
from django.contrib.auth.models import User

class Expense(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    date = models.DateField()
    category = models.CharField(max_length=50)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    description = models.TextField(blank=True)

    class Meta:
        app_label = '__main__'

from django.contrib import admin
admin.site.register(Expense)

# ==========================================
# 3. VIEWS (Pandas, ML, CRUD Logic)
# ==========================================
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required
import pandas as pd
import numpy as np
import io, base64, urllib
import matplotlib
matplotlib.use('Agg') # Enables background rendering of plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

@login_required
def index(request):
    if request.method == 'POST':
        Expense.objects.create(
            user=request.user,
            date=request.POST.get('date'),
            category=request.POST.get('category'),
            amount=request.POST.get('amount'),
            description=request.POST.get('description')
        )
        return redirect('index')
    
    return render(request, 'index.html', {'expenses': Expense.objects.filter(user=request.user).order_by('-date')})

@login_required
def delete_expense(request, id):
    expense = get_object_or_404(Expense, id=id, user=request.user)
    expense.delete()
    return redirect('index')

@login_required
def analysis(request):
    expenses = Expense.objects.filter(user=request.user).values()
    if not expenses:
        return render(request, 'analysis.html', {'error': 'No data available. Add expenses first.'})

    df = pd.DataFrame(expenses)
    df['amount'] = df['amount'].astype(float) # Fixes Decimal TypeError for Seaborn/Sklearn
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')
    
    # 3a. Pandas Output
    monthly_totals = df.groupby('month')['amount'].sum()
    avg_per_category = df.groupby('category')['amount'].mean()
    
    # 3b. Seaborn/Matplotlib Visualizations
    plt.figure(figsize=(6, 4))
    sns.barplot(x=avg_per_category.index, y=avg_per_category.values, hue=avg_per_category.index, legend=False, palette="viridis")
    plt.title('Average Spending per Category')
    plt.tight_layout()
    buf_bar = io.BytesIO()
    plt.savefig(buf_bar, format='png')
    buf_bar.seek(0)
    chart_bar = urllib.parse.quote(base64.b64encode(buf_bar.read()))
    plt.close()

    plt.figure(figsize=(6, 4))
    monthly_totals.plot(kind='line', marker='o', color='b')
    plt.title('Monthly Spending Trend')
    plt.tight_layout()
    buf_line = io.BytesIO()
    plt.savefig(buf_line, format='png')
    buf_line.seek(0)
    chart_line = urllib.parse.quote(base64.b64encode(buf_line.read()))
    plt.close()

    # 3c. Scikit-learn Feature scaling & Regression Prediction
    pred_str = "Need at least 2 months of data to predict next month."
    df_monthly = df.groupby('month')['amount'].sum().reset_index()
    if len(df_monthly) >= 2:
        df_monthly['month_num'] = np.arange(len(df_monthly))
        X = df_monthly[['month_num']]
        y = df_monthly['amount']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        pred_val = model.predict(scaler.transform([[len(df_monthly)]]))[0]
        pred_str = f"${round(pred_val, 2)}"

    return render(request, 'analysis.html', {
        'avg_per_category': avg_per_category.to_dict(),
        'chart_bar': chart_bar,
        'chart_line': chart_line,
        'prediction': pred_str
    })

def signup_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('index')
    else:
        form = UserCreationForm()
    return render(request, 'signup.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('index')
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('login')

@login_required
def profile_view(request):
    user_expenses = Expense.objects.filter(user=request.user)
    total_spent = sum(e.amount for e in user_expenses)
    return render(request, 'profile.html', {
        'total_spent': total_spent,
        'expense_count': user_expenses.count()
    })

# ==========================================
# 4. URLs (Routing)
# ==========================================
from django.urls import path

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', index, name='index'),
    path('delete/<int:id>/', delete_expense, name='delete'),
    path('analysis/', analysis, name='analysis'),
    path('signup/', signup_view, name='signup'),
    path('login/', login_view, name='login'),
    path('logout/', logout_view, name='logout'),
    path('profile/', profile_view, name='profile'),
]

# ==========================================
# 5. SERVER ENTRY POINT
# ==========================================
if __name__ == '__main__':
    from django.db import connection
    
    # 5a. Auto Migrate auth/session tables if runserver
    if len(sys.argv) > 1 and sys.argv[1] == 'runserver':
        from django.core.management import call_command
        call_command('migrate', interactive=False)

    # 5b. Auto-create Expense database tables
    with connection.schema_editor() as schema_editor:
        try:
            schema_editor.create_model(Expense)
        except Exception:
            try:
                with connection.cursor() as cursor:
                    cursor.execute("ALTER TABLE __main___expense ADD COLUMN user_id INTEGER REFERENCES auth_user(id)")
            except Exception:
                pass # Already altered

    # 5c. Start server
    execute_from_command_line(sys.argv)
