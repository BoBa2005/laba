# Generated by Django 4.2.7 on 2023-11-30 22:46

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Record',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('title', models.CharField(max_length=255)),
                ('keyword', models.CharField(max_length=255)),
                ('publication', models.CharField(max_length=255)),
                ('resource', models.CharField(max_length=255)),
                ('content', models.TextField()),
                ('date_published', models.DateField()),
            ],
        ),
    ]